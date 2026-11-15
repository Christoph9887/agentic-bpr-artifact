"""Phase-3 handlers added by `mabpr`, and their registration into Phase-3."""

from __future__ import annotations

_REGISTERED = False

def register() -> dict:
    """Add mabpr's handlers to Phase-3's dispatch table. Returns what it added."""
    global _REGISTERED
    from mabpr.foundation.transform import phase3_orchestrator as orch

    from mabpr.transform import (ac_redesign, ct_redesign, loop_redesign,
                                 ni_redesign, spc_redesign,
                                 triage_redesign, xr_redesign)

    added = {
        "Extra Resources": (xr_redesign.apply_extra_resources,
                            xr_redesign.adapt_params_after_xr, False),

        "Activity Composition": (ac_redesign.apply_activity_composition,
                                 ac_redesign.adapt_params_after_ac, True),

        "Specialize": (spc_redesign.apply_specialize,
                       spc_redesign.adapt_params_after_spc, False),

        "Case Types": (ct_redesign.apply_case_types,
                       ct_redesign.adapt_params_after_ct, False),

        "Contact Reduction": (loop_redesign.apply_loop_reduction,
                              loop_redesign.adapt_params_after_loop, False),
        "Interfacing": (loop_redesign.apply_loop_reduction,
                        loop_redesign.adapt_params_after_loop, False),

        "Triage": (triage_redesign.apply_triage,
                   triage_redesign.adapt_params_after_triage, False),

        "Numerical Involvement": (ni_redesign.apply_numerical_involvement,
                                  ni_redesign.adapt_params_after_ni, False),
    }
    for name, entry in added.items():
        orch.DISPATCH.setdefault(name, entry)
    _REGISTERED = True
    return added
