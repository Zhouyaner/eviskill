import random
from typing import Any, Dict, Iterable, List, Optional


def get_task_id(item: Any) -> str:
    if isinstance(item, dict):
        raw = item.get("task_id") or item.get("id") or item.get("task")
        if isinstance(raw, dict):
            return str(raw.get("task_id") or raw.get("id") or raw)
        return str(raw)
    return str(item)


def get_task_ref(item: Any) -> Any:
    if isinstance(item, dict):
        dataset = str(item.get("dataset", "")).lower()
        if dataset in {"alfworld", "scienceworld", "science_world", "appworld"}:
            return dict(item)
        if item.get("gamefile"):
            return dict(item)
        if any(key in item for key in ("variation_idx", "variation", "simplification", "simplification_str")):
            return dict(item)
    return get_task_id(item)


def get_task_family(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("task_family") or item.get("family") or item.get("task_type") or "other")
    return "other"


def _variation_key(item: Any) -> str:
    if isinstance(item, dict):
        for key in ("variation_idx", "variation", "skillnet_index"):
            value = item.get(key)
            if value not in (None, ""):
                return f"{key}:{value}"
    return f"task_id:{get_task_id(item)}"


def sample_random_bundle(
    manifest: List[Dict[str, Any]],
    bundle_size: int,
    *,
    rng: Optional[random.Random] = None,
    low_reward_task_ids: Optional[Iterable[str]] = None,
    ensure_half_low_reward: bool = False,
) -> List[str]:
    rng = rng or random
    bundle_size = min(int(bundle_size), len(manifest))
    if bundle_size <= 0:
        return []

    all_ids = [get_task_id(x) for x in manifest]
    low_set = {str(x) for x in low_reward_task_ids or []}

    if not ensure_half_low_reward or not low_set:
        chosen = rng.sample(list(manifest), bundle_size)
        return [get_task_ref(x) for x in chosen]

    low_pool = [tid for tid in all_ids if tid in low_set]
    other_pool = [tid for tid in all_ids if tid not in low_set]
    need_low = min(len(low_pool), bundle_size // 2)

    chosen = rng.sample(low_pool, need_low) if need_low else []
    remaining = bundle_size - len(chosen)
    candidates = [tid for tid in other_pool if tid not in chosen]
    if len(candidates) < remaining:
        candidates = [tid for tid in all_ids if tid not in chosen]
    chosen.extend(rng.sample(candidates, min(remaining, len(candidates))))
    rng.shuffle(chosen)
    by_id = {get_task_id(x): x for x in manifest}
    return [get_task_ref(by_id.get(tid, tid)) for tid in chosen]


def _family_epoch_order(
    manifest: List[Dict[str, Any]],
    *,
    seed: int,
    epoch: int,
) -> List[Any]:
    family_to_items: Dict[str, List[Any]] = {}
    for item in manifest:
        family_to_items.setdefault(get_task_family(item), []).append(item)
    if not family_to_items:
        return []

    rng = random.Random(int(seed) + int(epoch) * 1000)
    family_order = sorted(family_to_items)
    rng.shuffle(family_order)

    pools: Dict[str, List[Any]] = {}
    for family in family_order:
        pool = list(family_to_items[family])
        rng.shuffle(pool)
        pools[family] = pool

    ordered: List[Any] = []
    active = list(family_order)
    while active:
        next_active: List[str] = []
        for family in active:
            pool = pools.get(family) or []
            if not pool:
                continue
            ordered.append(pool.pop())
            if pool:
                next_active.append(family)
        active = next_active
    return ordered


def sample_family_epoch_bundle(
    manifest: List[Dict[str, Any]],
    bundle_size: int,
    *,
    seed: int,
    round_idx: int,
) -> List[Any]:
    """Return a deterministic family-balanced epoch bundle.

    Each epoch shuffles items within every task family, shuffles the family
    order, then interleaves families round-robin before slicing into bundles.
    This keeps short runs from being dominated by large task families while
    still making a full pass over the provided manifest before the next epoch.
    """
    manifest = list(manifest)
    if not manifest:
        return []
    bundle_size = min(int(bundle_size), len(manifest))
    if bundle_size <= 0:
        return []

    bundles_per_epoch = max(1, (len(manifest) + bundle_size - 1) // bundle_size)
    epoch = max(0, int(round_idx)) // bundles_per_epoch
    bundle_in_epoch = max(0, int(round_idx)) % bundles_per_epoch
    ordered = _family_epoch_order(manifest, seed=seed, epoch=epoch)
    start = bundle_in_epoch * bundle_size
    return [get_task_ref(x) for x in ordered[start : start + bundle_size]]


def sample_shuffle_epoch_bundle(
    manifest: List[Dict[str, Any]],
    bundle_size: int,
    *,
    seed: int,
    round_idx: int,
    epoch_seed_base: int = 1,
) -> List[Any]:
    """Return one bundle from a plain shuffled epoch over the manifest.

    This mirrors SkillOpt's split-backed batch planning: shuffle all train items
    once per epoch, then take contiguous batch_size slices. Unlike
    family_epoch, this does not round-robin or otherwise balance task families
    inside a bundle.
    """
    manifest = list(manifest)
    if not manifest:
        return []
    bundle_size = min(int(bundle_size), len(manifest))
    if bundle_size <= 0:
        return []

    bundles_per_epoch = max(1, (len(manifest) + bundle_size - 1) // bundle_size)
    # SkillOPT numbers epochs from one when deriving its split-backed shuffle
    # seed. epoch_seed_base=0 exists only to resume runs created before this
    # convention was aligned; new runs always use the default one-based plan.
    epoch = max(0, int(round_idx)) // bundles_per_epoch + int(epoch_seed_base)
    bundle_in_epoch = max(0, int(round_idx)) % bundles_per_epoch
    ordered = list(manifest)
    random.Random(int(seed) + int(epoch) * 1000).shuffle(ordered)
    start = bundle_in_epoch * bundle_size
    return [get_task_ref(x) for x in ordered[start : start + bundle_size]]


def mixed_family_epoch_bundle_mode(round_idx: int, family_group_frequency: int = 2) -> str:
    frequency = int(family_group_frequency)
    if frequency <= 0:
        return "diverse"
    if frequency == 1:
        return "same_family"
    return "same_family" if (max(0, int(round_idx)) + 1) % frequency == 0 else "diverse"


def _same_family_round_index(round_idx: int, family_group_frequency: int) -> int:
    frequency = int(family_group_frequency)
    if frequency <= 1:
        return max(0, int(round_idx))
    return max(0, int(round_idx)) // frequency


def _same_family_items_for_round(
    manifest: List[Dict[str, Any]],
    bundle_size: int,
    *,
    seed: int,
    same_round_idx: int,
) -> List[Any]:
    family_to_items: Dict[str, List[Any]] = {}
    for item in manifest:
        family_to_items.setdefault(get_task_family(item), []).append(item)
    if not family_to_items:
        return []

    families = sorted(family_to_items)
    family_epoch = max(0, int(same_round_idx)) // len(families)
    family_in_epoch = max(0, int(same_round_idx)) % len(families)
    family_order = list(families)
    random.Random(int(seed) + family_epoch * 1000).shuffle(family_order)
    family = family_order[family_in_epoch]
    items = list(family_to_items.get(family) or [])
    if not items:
        return []

    family_index = families.index(family)
    bundles_per_family_epoch = max(1, (len(items) + bundle_size - 1) // bundle_size)
    item_epoch = family_epoch // bundles_per_family_epoch
    item_bundle_in_epoch = family_epoch % bundles_per_family_epoch
    rng = random.Random(int(seed) + 100000 + family_index * 1009 + item_epoch * 1000)
    ordered = list(items)
    rng.shuffle(ordered)
    start = item_bundle_in_epoch * bundle_size
    cyclic = ordered[start:] + ordered[:start]

    selected: List[Any] = []
    selected_ids = set()
    seen_variations = set()
    for prefer_distinct in (True, False):
        for item in cyclic:
            if len(selected) >= bundle_size:
                return selected
            task_id = get_task_id(item)
            if task_id in selected_ids:
                continue
            variation = _variation_key(item)
            if prefer_distinct and variation in seen_variations:
                continue
            selected.append(item)
            selected_ids.add(task_id)
            seen_variations.add(variation)
    return selected


def sample_mixed_family_epoch_bundle(
    manifest: List[Dict[str, Any]],
    bundle_size: int,
    *,
    seed: int,
    round_idx: int,
    family_group_frequency: int = 2,
) -> List[Any]:
    """Return a deterministic mix of diverse and same-family epoch bundles."""
    manifest = list(manifest)
    if not manifest:
        return []
    bundle_size = min(int(bundle_size), len(manifest))
    if bundle_size <= 0:
        return []

    mode = mixed_family_epoch_bundle_mode(round_idx, family_group_frequency)
    if mode == "diverse":
        return sample_family_epoch_bundle(
            manifest,
            bundle_size,
            seed=seed,
            round_idx=round_idx,
        )

    selected = _same_family_items_for_round(
        manifest,
        bundle_size,
        seed=seed,
        same_round_idx=_same_family_round_index(round_idx, family_group_frequency),
    )
    selected_ids = {get_task_id(item) for item in selected}
    if len(selected) < bundle_size:
        fill = sample_family_epoch_bundle(
            manifest,
            bundle_size,
            seed=seed,
            round_idx=round_idx,
        )
        by_id = {get_task_id(item): item for item in manifest}
        for item in fill:
            task_id = get_task_id(item)
            if task_id in selected_ids:
                continue
            selected.append(by_id.get(task_id, item))
            selected_ids.add(task_id)
            if len(selected) >= bundle_size:
                break
    return [get_task_ref(x) for x in selected[:bundle_size]]


def sample_balanced_by_skill_count(
    manifest: List[Dict[str, Any]],
    bundle_size: int,
    *,
    skill_counts: Dict[str, int],
    rng: Optional[random.Random] = None,
) -> List[str]:
    rng = rng or random
    bundle_size = min(int(bundle_size), len(manifest))
    if bundle_size <= 0:
        return []

    family_to_items: Dict[str, List[Any]] = {}
    for item in manifest:
        family_to_items.setdefault(get_task_family(item), []).append(item)

    if not family_to_items:
        return []

    counts = {
        family: int(skill_counts.get(family, 0) or 0)
        for family in family_to_items
    }
    if len(set(counts.values())) <= 1:
        return sample_random_bundle(manifest, bundle_size, rng=rng)

    chosen: List[Any] = []
    chosen_ids = set()
    count_levels = sorted(set(counts.values()))
    for count in count_levels:
        pools = {
            family: [item for item in items if get_task_id(item) not in chosen_ids]
            for family, items in family_to_items.items()
            if counts.get(family, 0) == count
        }
        pools = {family: items for family, items in pools.items() if items}
        if not pools:
            continue
        while pools and len(chosen) < bundle_size:
            families = list(pools)
            rng.shuffle(families)
            for family in families:
                if len(chosen) >= bundle_size:
                    break
                pool = pools.get(family) or []
                if not pool:
                    pools.pop(family, None)
                    continue
                idx = rng.randrange(len(pool))
                item = pool.pop(idx)
                chosen.append(item)
                chosen_ids.add(get_task_id(item))
                if not pool:
                    pools.pop(family, None)

    if len(chosen) < bundle_size:
        pool = [x for x in manifest if get_task_id(x) not in chosen_ids]
        remaining = min(bundle_size - len(chosen), len(pool))
        if remaining > 0:
            chosen.extend(rng.sample(pool, remaining))

    rng.shuffle(chosen)
    return [get_task_ref(x) for x in chosen]


def low_reward_ids_from_logs(rows: Iterable[Dict[str, Any]], threshold: float = 1.0) -> List[str]:
    ids = []
    for row in rows:
        task_id = row.get("task_id")
        reward = row.get("final_reward", row.get("score"))
        if task_id is not None and reward is not None and float(reward) < threshold:
            ids.append(str(task_id))
    return ids


def limit_manifest(manifest: List[Dict[str, Any]], limit_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
    if limit_tasks is None or int(limit_tasks) <= 0:
        return manifest
    return list(manifest)[: int(limit_tasks)]
