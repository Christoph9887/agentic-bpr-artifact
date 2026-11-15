"""HI-Graph entries for the heuristics `mabpr` adds. `foundation` stays untouched."""

from __future__ import annotations

from mabpr.foundation.knowledge import phase2_hi_graph as hg

NEW_HEURISTICS: tuple = ("Extra Resources", "Activity Composition", "Specialize",
                         "Case Types", "Contact Reduction", "Interfacing",
                         "Triage", "Numerical Involvement")

_SOURCE = "prior work (reference withheld for double-blind review)"

def _register_names() -> None:
    """Make the new names valid for the validators and for the tool enum."""
    missing = tuple(h for h in NEW_HEURISTICS if h not in hg.HEURISTICS)
    if missing:
        hg.HEURISTICS = tuple(hg.HEURISTICS) + missing

def _impact_vectors() -> list:
    """One Devil's Quadrangle vector per new heuristic, quoting the spec."""
    IV = hg.ImpactVector
    return [
        IV(heuristic="Extra Resources",

           time="strong_positive",

           cost="strong_negative",
           quality="neutral",

           flexibility="weak_positive",
           rationale=("Adding servers to a saturated activity shortens the queue "
                      "directly, and pays for it in salaried capacity that is idle "
                      "whenever demand dips. The only heuristic here whose cost sign "
                      "is strongly negative — it buys time with money rather than by "
                      "removing work."),
           source=_SOURCE),
        IV(heuristic="Activity Composition",

           time="weak_positive",
           cost="weak_positive",
           quality="neutral",

           flexibility="weak_negative",
           rationale=("Merging consecutive small activities removes the gap and the "
                      "handover between them. Weak on both time and cost because "
                      "the work itself is unchanged — only the seam is removed — and "
                      "negative on flexibility because the merged task can no longer "
                      "be allocated to two different people."),
           source=_SOURCE),
        IV(heuristic="Specialize",

           time="weak_positive",
           cost="neutral",
           quality="weak_positive",

           flexibility="strong_negative",
           rationale=("Concentrating an activity on its fastest performers speeds it "
                      "up and makes it more consistent, at the cost of the pool: "
                      "strongly negative on flexibility because the activity can "
                      "afterwards be performed by only a handful of named people. "
                      "⚠️ The time gain is a SERVICE-time gain; whether the smaller "
                      "pool queues is measured, never predicted."),
           source=_SOURCE),
        IV(heuristic="Case Types",

           time="strong_positive",

           cost="weak_negative",

           quality="weak_positive",

           flexibility="weak_negative",
           rationale=("Giving each case type the route its characteristics imply "
                      "removes work simple cases never needed, so time is the "
                      "strongest axis; the cost is running and maintaining more "
                      "than one variant of the process. ⚠️ Unlike the other nine, "
                      "this heuristic changes WHICH cases reach an activity rather "
                      "than how long it takes or who performs it — so its effect "
                      "arrives entirely through the mix, and no hours are claimed."),
           source=_SOURCE),
        IV(heuristic="Contact Reduction",

           time="strong_positive",
           cost="weak_positive",

           quality="weak_positive",

           flexibility="weak_negative",
           rationale=("Asking once instead of three times removes wall-clock waiting "
                      "the process spends on somebody outside it, which is why time "
                      "is the strongest axis and cost follows. The risk the spec "
                      "names is on quality-of-information rather than throughput: a "
                      "single consolidated contact must collect everything. ⚠️ The "
                      "saving is EXOGENOUS — the mined model has no construct for an "
                      "overnight or customer-side wait, so the number is a log "
                      "measurement reported beside the simulation, never added to "
                      "it, and it is arbitrated by `evidence/exo_ledger`."),
           source=_SOURCE),
        IV(heuristic="Interfacing",

           time="strong_positive",
           cost="weak_positive",
           quality="strong_positive",

           flexibility="weak_negative",
           rationale=("A standardised exchange stops information coming back wrong, "
                      "so the correction loop is entered less often: quality is "
                      "strongly positive because that is the mechanism, and time "
                      "follows because the rework and its wait go with it. ⚠️ Like "
                      "Contact Reduction the saving is EXOGENOUS and ledger-"
                      "arbitrated; unlike it, the cost side is an implementation "
                      "project outside the process, which no event log can price."),
           source=_SOURCE),
        IV(heuristic="Triage",

           time="strong_positive",

           cost="weak_negative",
           quality="strong_positive",

           flexibility="weak_negative",
           rationale=("Sizing the work to the case instead of to the average speeds "
                      "the simple ones and gives the complex ones room, so time and "
                      "quality are both strong. The cost is a process that now has "
                      "more than one version of the same step to run and maintain. "
                      "⚠️ Measured zero on both datasets: every activity x case-type "
                      "effort gap is inside the noise (|z| < 3) despite ratios up to "
                      "8.72x — heavy-tailed durations, the same trap Specialize hits."),
           source=_SOURCE),
        IV(heuristic="Numerical Involvement",

           time="strong_positive",
           cost="weak_positive",
           quality="weak_positive",

           flexibility="strong_negative",
           rationale=("Fewer hands on a case removes the coordination and the waiting "
                      "between them, so time leads and cost follows. Flexibility is "
                      "strongly negative for the same reason Case Assignment's is: "
                      "the work is tied to whoever started it. ⚠️ Measured zero on "
                      "both datasets — the LENGTH-CONTROLLED handover rate is only "
                      "0.28-0.43, so consecutive steps already stay with one person "
                      "and the 4-6 participants per case are a consequence of case "
                      "LENGTH, not of fragmentation."),
           source=_SOURCE),
    ]

def _relations() -> list:
    """Pairwise priors for every pair involving a new heuristic."""

    TR, share = hg.TypeRelation, hg._cond_share_activity
    neutral = hg.ConditionEncoding(text="applies unconditionally",
                                   predicate=lambda a, b: True)

    def rel(a, b, kind, cond, why):
        return TR(heuristic_a=a, heuristic_b=b, relation_type=kind,
                  condition=cond, rationale=why, source=_SOURCE)

    out = [

        *[rel(h, h, "NEUTRAL", neutral,
              "a heuristic does not conflict with itself at the type level")
          for h in NEW_HEURISTICS],

        rel("Extra Resources", "Specialize", "HARD_CONFLICT", share(),
            "Directly opposed on the same activity: one widens the pool serving it, "
            "the other narrows it to the fastest few. Applying both to one activity "
            "leaves whichever ran last, so the plan's stated intent is not what gets "
            "built."),
        rel("Activity Elimination", "Extra Resources", "HARD_CONFLICT", share(),
            "Capacity added to an activity another candidate deletes is capacity for "
            "nothing; the params edit targets a task that no longer exists."),
        rel("Case Assignment", "Extra Resources", "WEAK_CONFLICT", share(),
            "Both change who serves an activity. CA merges pools, XR multiplies "
            "them, so together they can inflate a pool far beyond either's intent."),
        rel("Activity Composition", "Extra Resources", "WEAK_CONFLICT", share(),
            "Adding capacity to a task that is about to be merged away moves the "
            "capacity onto the composite, which is not what was sized."),
        *[rel(h, "Extra Resources", "WEAK_CONFLICT", share(), why)
          for h, why in (
              ("Parallelism",
               "Parallelism makes two activities run at once, so demand arrives at "
               "the pool concurrently rather than in sequence — the capacity XR "
               "sized for a sequential arrival pattern now faces a burst."),
              ("Resequencing",
               "Moving an activity changes WHEN its demand reaches the pool, which "
               "is the quantity XR's sizing is computed from."),
              ("Knock-out",
               "Reordering checks changes how OFTEN the activity downstream runs, so "
               "capacity sized on the observed volume may be sized for a volume that "
               "no longer occurs."),
              ("Exception",
               "An Exception moves the activity into a subprocess; the servers XR "
               "added follow it there and now serve a different population."),
              ("Case-based Work",
               "Both change the timing of work at the activity — one by adding "
               "servers, the other by removing the wait for a batch to fill — so on "
               "a shared activity their effects are not separable."))],

        rel("Activity Elimination", "Case Types", "HARD_CONFLICT", share(),
            "The conditioned branch leads to exactly one task, and AE deletes it. "
            "The rule then routes a case type to nothing, so the redesign that was "
            "measured is not the one that gets built."),
        rel("Activity Composition", "Case Types", "HARD_CONFLICT", share(),
            "Composition deletes one activity and renames the other, so the single "
            "task the conditioned path was chosen for is no longer the task the "
            "path leads to."),
        rel("Case Types", "Exception", "HARD_CONFLICT", share(),
            "An Exception lifts activities into a subprocess and rewires the flow "
            "around the divergence point, so the path id the rule is attached to "
            "may no longer exist or may no longer reach the claimed activity."),
        rel("Case Types", "Parallelism", "HARD_CONFLICT", share(),
            "Parallelism replaces the exclusive choice with a split that takes both "
            "branches. A branch_rules condition on it decides nothing — the case "
            "type would traverse the branch it was routed away from anyway."),
        rel("Case Types", "Resequencing", "HARD_CONFLICT", share(),
            "Moving an activity changes which gateway precedes it, so the "
            "conditioned path no longer routes to the activity the log evidence "
            "was measured on."),
        rel("Case Types", "Knock-out", "WEAK_CONFLICT", share(),
            "Both change how often downstream work runs — Knock-out by reordering "
            "the checks that end a case early, Case Types by routing whole "
            "populations elsewhere. The visit shares CT measured were observed "
            "under the original check order."),
        rel("Case Assignment", "Case Types", "WEAK_CONFLICT", share(),
            "CT changes WHICH cases arrive at an activity, CA changes WHO serves "
            "it. A pool merged for the observed population now serves a different "
            "one, so neither candidate's effect on a shared activity is separable "
            "from the other's."),
        rel("Case Types", "Case-based Work", "WEAK_CONFLICT", share(),
            "CBW's batching depends on how work accumulates at an activity, and "
            "routing a case type away from it changes both the rate and the mix "
            "that fills the batch."),
        rel("Case Types", "Extra Resources", "WEAK_CONFLICT", share(),
            "XR sizes capacity from the volume observed at the activity; CT changes "
            "that volume by construction, so the sizing is computed for a demand "
            "that no longer arrives."),
        rel("Case Types", "Specialize", "WEAK_CONFLICT", share(),
            "SPC picks the smallest subset that can still carry the activity's "
            "workload, and CT changes that workload — the surviving specialists may "
            "be sized for a population the redesign has routed away."),

        rel("Case Assignment", "Numerical Involvement", "HARD_CONFLICT", share(),
            "Both widen who may perform the same activities. Applied together the "
            "pool is merged twice and neither candidate's effect is its own."),
        rel("Numerical Involvement", "Specialize", "HARD_CONFLICT", share(),
            "Directly opposed: NI lets everyone in the run perform every step, SPC "
            "restricts the step to its fastest few. Whichever runs last wins."),
        rel("Activity Elimination", "Numerical Involvement", "HARD_CONFLICT", share(),
            "A run of steps to put in one pair of hands cannot include a step that "
            "has been deleted."),
        rel("Activity Composition", "Numerical Involvement", "HARD_CONFLICT", share(),
            "Composition already merges two of the run's activities into one node "
            "under a single resource, which is the same intent applied structurally."),
        rel("Exception", "Numerical Involvement", "HARD_CONFLICT", share(),
            "An Exception moves part of the run into a subprocess, so the stretch is "
            "no longer contiguous and the widened allocation spans two scopes."),
        rel("Numerical Involvement", "Extra Resources", "WEAK_CONFLICT", share(),
            "Both enlarge a pool — one by letting existing people do more, the other "
            "by multiplying servers — so together they can inflate it past either's "
            "intent."),
        rel("Contact Reduction", "Numerical Involvement", "WEAK_CONFLICT", share(),
            "Consolidating repeat contacts changes how many steps the run actually "
            "has, which is the stretch being put in one pair of hands."),
        rel("Interfacing", "Numerical Involvement", "WEAK_CONFLICT", share(),
            "Fewer corrections shortens the run and changes the handover rate it was "
            "selected on."),
        rel("Case Types", "Numerical Involvement", "WEAK_CONFLICT", share(),
            "Routing a case type away changes which cases traverse the run, and the "
            "fragmentation was measured over the old population."),
        rel("Case-based Work", "Numerical Involvement", "WEAK_CONFLICT", share(),
            "Removing the batch wait changes when each step of the run is reached, "
            "and with it whether one person is still free to take the next."),
        rel("Numerical Involvement", "Parallelism", "WEAK_CONFLICT", share(),
            "Steps made concurrent cannot be carried sequentially by one person, so "
            "Parallelism removes part of what the widening is for."),
        rel("Numerical Involvement", "Resequencing", "WEAK_CONFLICT", share(),
            "Reordering breaks the contiguity of the run the handover rate was "
            "measured over."),
        rel("Knock-out", "Numerical Involvement", "WEAK_CONFLICT", share(),
            "Rejecting cases earlier changes how many of them travel the whole run."),
        rel("Numerical Involvement", "Triage", "WEAK_CONFLICT", share(),
            "Triage splits one step of the run into per-type versions, so the "
            "stretch one person was to carry now has a branch in it."),

        rel("Activity Elimination", "Triage", "HARD_CONFLICT", share(),
            "There is nothing to split into per-type versions once the activity is "
            "deleted."),
        rel("Activity Composition", "Triage", "HARD_CONFLICT", share(),
            "Composition fuses the activity into a neighbour, so the node Triage "
            "would clone no longer exists under the name that was measured."),
        rel("Exception", "Triage", "HARD_CONFLICT", share(),
            "An Exception absorbs the activity into a generated subprocess; the "
            "clones and their split would have to be built inside it."),
        rel("Case Types", "Triage", "WEAK_CONFLICT", share(),
            "Both route by the same case attribute, one at the process level and one "
            "at a single activity. Applied together the type is split twice, and the "
            "effort profile Triage measured was observed under the OLD routing."),
        rel("Specialize", "Triage", "WEAK_CONFLICT", share(),
            "Both re-cut the same service time — SPC by who performs it, Triage by "
            "which case it is. On a shared activity their fitted distributions are "
            "estimated from overlapping subsets of the same executions."),
        rel("Extra Resources", "Triage", "WEAK_CONFLICT", share(),
            "XR sizes capacity for one pooled activity; after the split the demand "
            "arrives at two nodes and the sizing no longer describes either."),
        rel("Case Assignment", "Triage", "WEAK_CONFLICT", share(),
            "CA merges a pool to serve the activity; the split gives that pool two "
            "activities to serve instead of one."),
        rel("Contact Reduction", "Triage", "WEAK_CONFLICT", share(),
            "Consolidating repeat contacts changes how many executions of the "
            "activity each type contributes, which is the population Triage fitted "
            "its per-type durations on."),
        rel("Interfacing", "Triage", "WEAK_CONFLICT", share(),
            "Fewer corrections changes the mix of executions the per-type "
            "distributions were estimated from."),
        rel("Case-based Work", "Triage", "WEAK_CONFLICT", share(),
            "Removing the batch wait changes when each type reaches the activity, so "
            "the two versions no longer face the arrival pattern that was measured."),
        rel("Parallelism", "Triage", "WEAK_CONFLICT", share(),
            "Running the activity concurrently changes how much of each type's effort "
            "sits on the critical path, which is the whole quantity Triage acts on."),
        rel("Resequencing", "Triage", "WEAK_CONFLICT", share(),
            "Moving the activity changes which case attributes are known by the time "
            "it runs, and the split has to be decidable at that point."),
        rel("Knock-out", "Triage", "WEAK_CONFLICT", share(),
            "Rejecting cases earlier removes part of one type's population, so the "
            "version sized for it faces a different load."),

        rel("Contact Reduction", "Interfacing", "HARD_CONFLICT", share(),
            "Both re-rate the same rework gateway: one because the exchange is "
            "repeated, the other because information comes back wrong. Applying "
            "both to one activity leaves whichever ran last, and the exogenous "
            "hours would be banked twice."),

        rel("Activity Elimination", "Contact Reduction", "HARD_CONFLICT", share(),
            "The loop probability being lowered belongs to an activity AE deletes; "
            "after the deletion there is no gateway path left to re-rate."),
        rel("Activity Composition", "Contact Reduction", "HARD_CONFLICT", share(),
            "Composition renames one activity and deletes the other, so the task the "
            "rework loop returns to is no longer the task that was measured."),
        rel("Contact Reduction", "Exception", "HARD_CONFLICT", share(),
            "An Exception lifts the activity into a subprocess and rewires the flow "
            "around it, so the loop path this candidate re-rates may no longer exist."),
        rel("Case Types", "Contact Reduction", "HARD_CONFLICT", share(),
            "Case Types attaches a hard branch_rules condition to a gateway; a "
            "condition overrides the probabilities Contact Reduction edits, so on a "
            "shared gateway one of the two silently does nothing."),
        rel("Contact Reduction", "Case-based Work", "WEAK_CONFLICT", share(),
            "Both claim an EXOGENOUS wall-clock wait at the same activity — one the "
            "wait for a batch to fill, the other the wait on an outside party. The "
            "exo ledger keeps them from banking the same hours, but their measured "
            "effects on a shared activity are still not separable."),
        rel("Case Assignment", "Contact Reduction", "WEAK_CONFLICT", share(),
            "CA changes who performs the contact; consolidating the contacts changes "
            "how often they perform it, so the pool CA sized serves a different load."),
        rel("Contact Reduction", "Parallelism", "WEAK_CONFLICT", share(),
            "Parallelism changes when the contact is reached, which changes how much "
            "of the idle wait after it overlaps other work rather than delaying the "
            "case."),
        rel("Contact Reduction", "Resequencing", "WEAK_CONFLICT", share(),
            "Moving the activity changes which gateway precedes it, and with it the "
            "loop this candidate measured."),
        rel("Contact Reduction", "Knock-out", "WEAK_CONFLICT", share(),
            "Knock-out changes how many cases reach the contact at all, so the repeat "
            "share the consolidation is sized on no longer describes the population."),
        rel("Contact Reduction", "Extra Resources", "WEAK_CONFLICT", share(),
            "XR sizes capacity from the observed volume at the activity; consolidating "
            "repeat contacts removes part of that volume."),
        rel("Contact Reduction", "Specialize", "WEAK_CONFLICT", share(),
            "SPC picks the smallest subset that can still carry the activity's "
            "workload, and fewer repeat contacts is less workload."),

        rel("Activity Elimination", "Interfacing", "HARD_CONFLICT", share(),
            "The correction loop being re-rated hangs off an activity AE deletes."),
        rel("Activity Composition", "Interfacing", "HARD_CONFLICT", share(),
            "The task the correction loop returns to is renamed or removed by the "
            "merge."),
        rel("Exception", "Interfacing", "HARD_CONFLICT", share(),
            "Exception rewires the divergence point the correction loop runs through."),
        rel("Case Types", "Interfacing", "HARD_CONFLICT", share(),
            "A branch_rules condition overrides the probability Interfacing edits, so "
            "on a shared gateway one of the two silently does nothing."),
        rel("Case-based Work", "Interfacing", "WEAK_CONFLICT", share(),
            "Both claim an exogenous wall-clock wait at the same activity; the ledger "
            "arbitrates the hours, but the measured effects stay entangled."),
        rel("Case Assignment", "Interfacing", "WEAK_CONFLICT", share(),
            "Fewer corrections is less work for the pool CA merged, so the pool is "
            "sized for a load that no longer arrives."),
        rel("Interfacing", "Parallelism", "WEAK_CONFLICT", share(),
            "Running the submission concurrently with other work changes how much of "
            "the correction wait is on the critical path."),
        rel("Interfacing", "Resequencing", "WEAK_CONFLICT", share(),
            "Reordering changes which activities precede the submission, and the "
            "rework this candidate counts is measured as work done BEFORE it."),
        rel("Interfacing", "Knock-out", "SYNERGY", share(),
            "Mutually reinforcing: Knock-out rejects unsuitable cases before effort is "
            "spent, Interfacing stops suitable ones bouncing back for correction. Both "
            "cut the population that reaches the expensive part of the process, and "
            "neither removes the other's mechanism."),
        rel("Extra Resources", "Interfacing", "WEAK_CONFLICT", share(),
            "Capacity sized on the observed volume faces a smaller volume once the "
            "correction loop fires less often."),
        rel("Interfacing", "Specialize", "WEAK_CONFLICT", share(),
            "The subset SPC sized to carry the activity's workload carries less of it "
            "once corrections stop recurring."),

        rel("Activity Composition", "Activity Elimination", "HARD_CONFLICT", share(),
            "Both rewrite the same region: a composite cannot be formed from an "
            "activity that has been deleted, and deleting half a composite leaves "
            "the merged service time attached to work that is gone."),
        rel("Activity Composition", "Exception", "HARD_CONFLICT", share(),
            "Exception moves activities into a subprocess, which breaks the direct "
            "flow a composition requires; whichever applies second finds its target "
            "no longer adjacent."),
        rel("Activity Composition", "Parallelism", "HARD_CONFLICT", share(),
            "Contradictory by construction: Parallelism splits a sequence so two "
            "activities run at once, Composition fuses them into one that cannot."),
        rel("Activity Composition", "Resequencing", "HARD_CONFLICT", share(),
            "Reordering breaks the adjacency the merge depends on."),
        rel("Activity Composition", "Knock-out", "WEAK_CONFLICT", share(),
            "Merging a check into its neighbour removes the early exit the knock-out "
            "ordering exists to create."),
        rel("Activity Composition", "Case Assignment", "SYNERGY", share(),
            "Mutually reinforcing: the composite must be performed by ONE resource, "
            "which is exactly the pooling Case Assignment arranges."),
        rel("Activity Composition", "Case-based Work", "WEAK_CONFLICT", share(),
            "If the batched activity is one half of the composite, the activity CBW "
            "targets stops existing under that name once the merge lands — the two "
            "are describing edits to the same piece of work."),
        rel("Activity Composition", "Specialize", "WEAK_CONFLICT", share(),
            "The spec names 'potential loss of specialization' as Composition's own "
            "trade-off: a merged task needs someone who can do both halves, which "
            "is the opposite of concentrating each half on its own specialist."),

        rel("Activity Elimination", "Specialize", "HARD_CONFLICT", share(),
            "Narrowing the allocation of an activity another candidate removes edits "
            "a task that no longer exists."),
        rel("Case Assignment", "Specialize", "HARD_CONFLICT", share(),
            "Exact opposites on a shared activity: CA widens the pool so one person "
            "can carry a case across steps, Specialize restricts it to named fast "
            "performers. Order decides which survives."),
        rel("Exception", "Specialize", "WEAK_CONFLICT", share(),
            "An exception branch pulls its activities into a subprocess, so a "
            "specialised allocation may end up applied to the isolated copy rather "
            "than to the mainstream cases it was sized for."),
        *[rel(h, "Specialize", "WEAK_CONFLICT", share(), why)
          for h, why in (
              ("Parallelism",
               "Running two activities at once while one of them has been narrowed to "
               "a handful of specialists puts concurrent demand on a pool that was "
               "just made smaller — the two edits pull against each other."),
              ("Resequencing",
               "Moving the activity changes when its demand lands on the specialist "
               "subset, which is what that subset was sized against."),
              ("Knock-out",
               "Reordering checks changes how often the specialised activity runs, so "
               "a subset sized on the observed volume may be sized for the wrong one."),
              ("Case-based Work",
               "Both change the timing of work at the activity — one by concentrating "
               "who does it, the other by removing the batch wait — so on a shared "
               "activity their effects are not separable."))],
    ]
    return out

def extend(graph):
    """Add the new heuristics to an ALREADY-BUILT graph. Idempotent."""

    _register_names()
    for iv in _impact_vectors():
        graph.impact_vectors.setdefault(iv.heuristic, iv)
    for tr in _relations():
        graph.type_relations.setdefault(
            tuple(sorted((tr.heuristic_a, tr.heuristic_b))), tr)
    return graph

_ORIGINAL_HEURISTICS: tuple = tuple(hg.HEURISTICS)
_ORIGINAL_BUILD = hg.build_hi_graph

def build_extended_hi_graph():
    """`build_hi_graph`, but returning the graph with mabpr's heuristics in it."""
    saved = hg.HEURISTICS
    hg.HEURISTICS = _ORIGINAL_HEURISTICS
    try:
        graph = _ORIGINAL_BUILD()
    finally:
        hg.HEURISTICS = saved
    return extend(graph)

hg.build_hi_graph = build_extended_hi_graph
