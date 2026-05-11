"""Prepare phase-wise train/test splits for the TIME curriculum.

Loads the raw ``phase{1,2,3}.json`` conversation files, injects system prompts
into a random 5% of examples, constructs 85/15 train/test splits, appends
replay data from prior phases, deduplicates, and writes the final splits to
``{data_dir}/phase{1,2,3}_{train,test}.json``.

Pre-generated splits are included in ``data/`` and can be used directly.
Run this script only if you want to regenerate them from the raw phase files.

Usage
-----
    python scripts/prepare_data.py
    python scripts/prepare_data.py --data-dir /path/to/data --seed 42
"""

import argparse
import copy
import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# System prompt pool
# ---------------------------------------------------------------------------
# A diverse set of 30 prompts that explain the <time> / <think> conventions.
# Injected into 5% of training conversations to provide grounding.

_SYSTEM_PROMPTS = [
    "You are an AI assistant. Every user message begins with a <time> tag showing the exact moment the turn occurs. Sometimes, there's only the timestamp with no text—that means time advanced without user input. Use <think>...</think> for your internal reasoning, notes, or meta-cognition; keep these hidden from the user.",
    "Each message starts with <time> to indicate when it happens. If there's no content after the timestamp, treat it as a time tick for you to process. Anything you want to figure out or plan should go inside <think>...</think>—these are just for your own thought process, not for the user to see.",
    "Whenever you see <time> at the top of a message, that's the timestamp. Sometimes it's followed by a blank message, which simply means time passed. Use <think>...</think> blocks for any private thoughts, calculations, or reflections you have—they're not shown to the user.",
    "Messages from the user always begin with a <time> tag. If nothing else is present, consider it a 'tick'—time has moved on. Your internal planning, steps, or meta-reasoning belong inside <think>...</think>; these are never visible to the user.",
    "You get turns that start with <time>. When the message is empty after the timestamp, it's just an automated tick so you can track time. When you need to reason or make plans, use <think>...</think> blocks for your private thoughts. Never show these blocks to the user.",
    "As an assistant, expect each message to start with <time> for the timestamp. Sometimes there's only a timestamp and nothing else—that's a time tick. Whenever you reason, process, or make internal notes, use <think>...</think>. These are for you alone, not the user.",
    "You are an AI that handles messages with a <time> tag up front. Blank messages after the timestamp mean just a tick—no new user input. For your internal logic, explanations, or calculations, always wrap them in <think>...</think>. Don't reveal these to the user.",
    "Every message has a <time> tag first. Sometimes there's no user text, just the time—that's your cue that time passed. Your behind-the-scenes thoughts and strategies belong in <think>...</think>; the user never sees these.",
    "Messages kick off with <time>, which is always the timestamp. If there's nothing after, it's just a time update. For anything you need to reason through or work out in private, put it in <think>...</think> blocks—these are never user-facing.",
    "You receive messages beginning with a <time> tag. When there's no message after the timestamp, it's an automated tick for time passing. Use <think>...</think> to hold your internal thoughts, notes, and reasoning steps; these are kept private.",
    "When you see a message with just a <time> tag and no content, that's a signal that time has advanced with no user activity. Your internal processing and reflections should go in <think>...</think>—this is your private workspace, not visible to the user.",
    "You are designed to interpret each turn's <time> tag as the timestamp. Sometimes, the only thing there is the time itself—no message, just a tick. Place any internal dialogue, meta-reasoning, or mental calculations inside <think>...</think> blocks. These are for your cognition only.",
    "User messages always start with <time>. If there's nothing after the timestamp, treat it as a clock tick. Make sure to keep all your reasoning or planning inside <think>...</think> blocks so it's never exposed to the user.",
    "Every message starts with a <time> marker. A blank message means time has moved on. If you need to work out anything or think through a problem, jot it down inside <think>...</think>—keep those thoughts private.",
    "Expect all user messages to start with <time> showing the current timestamp. When the message is empty after the timestamp, it's just a time step. Store your calculations and internal logic inside <think>...</think> tags, which are only for you.",
    "You are an AI assistant. Each message arrives with a <time> tag; sometimes, that's the only thing, marking a new time tick. Use <think>...</think> for your own mental notes, intermediate steps, and meta-reasoning. Don't show these to the user.",
    "Whenever you process a turn, check for <time>—it's the timestamp. If no message follows, take it as a cue for a silent time advance. Put all your reflections or planning into <think>...</think>—these thoughts are hidden from the user.",
    "Each user message comes prefixed with a <time> tag for timing. Empty content means only the time has changed. All internal deliberations, strategies, or meta-thinking should be within <think>...</think>, and never visible to the user.",
    "Messages you receive always have a <time> at the beginning, which may be followed by nothing—just a time update. Use <think>...</think> for all private reasoning, calculations, or step-by-step thoughts. The user won't see what's inside these blocks.",
    "You are an AI assistant. Every incoming turn has a <time> tag for the timestamp. Sometimes the message itself is blank, signaling only that time has advanced. Place your transient or meta-cognitive thoughts inside <think>...</think>; those blocks are strictly for your internal process and not for the user.",
    "System: Each user message includes a <time>...</time> tag marking the absolute timestamp of the turn. Sometimes, a message may contain only a timestamp, representing an automated 'tick'—a time advance with no user input, which you can use for temporal reasoning or agent reflection. As the assistant, use <think>...</think> blocks for your private, transient thoughts or meta-reasoning. These thoughts are not visible to the user and are for your own planning and cognition.",
    "Hey assistant! When you see <time>...</time> in a user message, that's the timestamp for that turn—it could be an actual question or just a time tick with no message, letting you keep track of the passage of time. For your internal thinking or reasoning steps, wrap them in <think>...</think> tags. Remember, those are just for you—not for the user to see!",
    "Note: Every user message has a <time> tag for timing. If the message is blank, it's just a time update—feel free to use it to reason about what happens as time goes by. When you need to think something through or plan, use <think>...</think>. These are your private thoughts only.",
    "Instructions: Treat <time>...</time> in user turns as absolute timestamps; blank messages mean time has advanced without input. For your own internal, private reasoning, use <think>...</think> blocks—never show these to the user.",
    "System Guidance: 1. Each user message begins with a <time> tag denoting the timestamp. 2. Sometimes, the user message will be empty—this is a clock tick for you to process time passing. 3. Use <think>...</think> for any internal or meta reasoning; these thoughts are not shared with the user, and may appear between your outward responses.",
    "Hi, just so you know: Whenever a message pops up, there's a <time> thingy up front—that's the date and time. Sometimes the message after the timestamp is empty (just a tick—time moves on). When you want to jot down your thoughts or plan ahead, use <think>...</think>—but keep it to yourself, that's just between us, not for the user!",
    "System Instructions: User inputs are always prefixed with a <time>...</time> timestamp, which can represent either a user-initiated message or an automated tick (empty message) to simulate the passage of time. As the assistant, encapsulate any transient, intermittent, or meta-cognitive reasoning in <think>...</think> blocks. These blocks are strictly for internal deliberation and should never be rendered to the user as part of the visible response.",
    "Reminder: User turns start with a <time> timestamp; sometimes that's all there is (an empty message means time passed—maybe you need to reflect or update your state). Use <think>...</think> whenever you need to process information, make plans, or reason in the background. These are private and never user-facing.",
    "Heads up! You'll always see <time> tags showing when each message happens. If there's nothing after the timestamp, it's just a tick to keep things moving along—think of it as the clock advancing. For your own mental notes, ideas, or calculations, use <think>...</think>. Keep those private—they're your behind-the-scenes thoughts.",
    "Protocol: - All user messages are prefixed by a <time>...</time> tag, which can indicate a user message or a blank automated tick. - When you, the assistant, need to reason, reflect, or plan, wrap those private thoughts in <think>...</think> blocks. These are for your use only and never appear in the user's view.",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate phase-wise train/test splits for the TIME curriculum. "
            "Pre-generated splits are already present in data/; run this only "
            "to regenerate them from raw phase files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing phase{1,2,3,4}.json and where splits will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Primary random seed (seed+1 is used for the Phase 1 replay into Phase 3).",
    )
    parser.add_argument(
        "--system-prompt-pct",
        type=float,
        default=0.05,
        help="Fraction of conversations that receive a system prompt injection.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.85,
        help="Train fraction for the 85/15 split.",
    )
    parser.add_argument(
        "--replay-pct",
        type=float,
        default=0.25,
        help="Fraction of a prior phase sampled for replay into the next phase.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def load_json(path: Path) -> list:
    """Load a JSON conversation file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list, path: Path) -> None:
    """Write a conversation list as a formatted JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(data):>6,} conversations → {path}")


def inject_system_prompts(
    conversations: list,
    prompts: list,
    rng: random.Random,
    pct: float = 0.05,
) -> list:
    """Insert a system prompt at the start of *pct* fraction of conversations.

    A deep copy is made so the original list is not modified.
    """
    data = copy.deepcopy(conversations)
    k = int(pct * len(data))
    selected = rng.sample(range(len(data)), k)
    for idx in selected:
        data[idx].insert(0, {"role": "system", "content": rng.choice(prompts)})
    return data


def split_train_test(
    data: list,
    rng: random.Random,
    train_ratio: float = 0.85,
) -> tuple[list, list]:
    """Shuffle and split *data* into train/test sets."""
    shuffled = data.copy()
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * train_ratio)
    return shuffled[:cut], shuffled[cut:]


def sample_for_replay(
    train: list,
    test: list,
    pct: float,
    rng: random.Random,
    train_ratio: float = 0.85,
) -> tuple[list, list]:
    """Sample *pct* of a phase (train + test combined) for replay.

    The sample is itself re-split at *train_ratio* to maintain train/test
    balance in the receiving phase.
    """
    combined = train + test
    rng.shuffle(combined)
    sample = combined[: int(len(combined) * pct)]
    cut = int(len(sample) * train_ratio)
    return sample[:cut], sample[cut:]


def merge_and_deduplicate(
    primary_train: list,
    primary_test: list,
    extra_trains: list[list],
    extra_tests: list[list],
) -> tuple[list, list]:
    """Merge replay data into a primary split and remove exact duplicates.

    Deduplication is performed by JSON-serialising each conversation and
    collecting into a set, so ordering within each conversation matters.
    """
    def _dedup(convs: list) -> list:
        seen: set = set()
        unique = []
        for conv in convs:
            key = json.dumps(conv, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique.append(conv)
        return unique

    merged_train = primary_train + sum(extra_trains, [])
    merged_test = primary_test + sum(extra_tests, [])
    return _dedup(merged_train), _dedup(merged_test)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Build all train/test splits."""
    args = get_args()
    data_dir = Path(args.data_dir)

    rng1 = random.Random(args.seed)       # primary seed (used for phases 1–3 replay)
    rng2 = random.Random(args.seed + 1)   # secondary seed (phase 1 → phase 3 replay)

    # ------------------------------------------------------------------ #
    # Load raw phase data
    # ------------------------------------------------------------------ #
    print("Loading raw phase data...")
    phase1_raw = load_json(data_dir / "phase1.json")
    phase2_raw = load_json(data_dir / "phase2.json")
    phase3_raw = load_json(data_dir / "phase3.json")
    print(
        f"  Phase 1: {len(phase1_raw):,} | "
        f"Phase 2: {len(phase2_raw):,} | "
        f"Phase 3: {len(phase3_raw):,} conversations"
    )

    # ------------------------------------------------------------------ #
    # Inject system prompts (~5% of each phase)
    # ------------------------------------------------------------------ #
    print(f"\nInjecting system prompts into {args.system_prompt_pct:.0%} of each phase...")
    phase1 = inject_system_prompts(phase1_raw, _SYSTEM_PROMPTS, rng1, args.system_prompt_pct)
    phase2 = inject_system_prompts(phase2_raw, _SYSTEM_PROMPTS, rng1, args.system_prompt_pct)
    phase3 = inject_system_prompts(phase3_raw, _SYSTEM_PROMPTS, rng1, args.system_prompt_pct)

    # ------------------------------------------------------------------ #
    # 85/15 train/test splits for each phase
    # ------------------------------------------------------------------ #
    print("\nBuilding 85/15 train/test splits...")
    p1_train, p1_test = split_train_test(phase1, rng1, args.train_ratio)
    p2_train, p2_test = split_train_test(phase2, rng1, args.train_ratio)
    p3_train, p3_test = split_train_test(phase3, rng1, args.train_ratio)

    # ------------------------------------------------------------------ #
    # Replay sampling
    # Phase 2 gets 25% of Phase 1 (rng1).
    # Phase 3 gets 25% of Phase 1 (rng2, different seed to avoid overlap)
    #         and 25% of Phase 2 (rng1).
    # ------------------------------------------------------------------ #
    print("\nSampling replay data...")
    p1_to_p2_train, p1_to_p2_test = sample_for_replay(
        p1_train, p1_test, args.replay_pct, rng1, args.train_ratio
    )
    p1_to_p3_train, p1_to_p3_test = sample_for_replay(
        p1_train, p1_test, args.replay_pct, rng2, args.train_ratio
    )
    p2_to_p3_train, p2_to_p3_test = sample_for_replay(
        p2_train, p2_test, args.replay_pct, rng1, args.train_ratio
    )

    # ------------------------------------------------------------------ #
    # Merge and deduplicate
    # ------------------------------------------------------------------ #
    print("\nMerging replay data and deduplicating...")
    p2_train_final, p2_test_final = merge_and_deduplicate(
        p2_train, p2_test,
        [p1_to_p2_train], [p1_to_p2_test],
    )
    p3_train_final, p3_test_final = merge_and_deduplicate(
        p3_train, p3_test,
        [p1_to_p3_train, p2_to_p3_train],
        [p1_to_p3_test, p2_to_p3_test],
    )

    print(
        f"  Phase 2 final — Train: {len(p2_train_final):,}, Test: {len(p2_test_final):,}"
    )
    print(
        f"  Phase 3 final — Train: {len(p3_train_final):,}, Test: {len(p3_test_final):,}"
    )

    # ------------------------------------------------------------------ #
    # Save all splits
    # ------------------------------------------------------------------ #
    print("\nSaving splits...")
    save_json(p1_train,       data_dir / "phase1_train.json")
    save_json(p1_test,        data_dir / "phase1_test.json")
    save_json(p2_train_final, data_dir / "phase2_train.json")
    save_json(p2_test_final,  data_dir / "phase2_test.json")
    save_json(p3_train_final, data_dir / "phase3_train.json")
    save_json(p3_test_final,  data_dir / "phase3_test.json")

    print("\nDone. All splits written to", data_dir)


if __name__ == "__main__":
    main()
