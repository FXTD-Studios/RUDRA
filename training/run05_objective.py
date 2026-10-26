"""Opt-in restoration objective: retain target supervision, remove residual shrinkage."""


def restoration_total(losses, region_focus=0.0, crushed_hold=0.0):
    if region_focus != 0 or crushed_hold != 0:
        raise ValueError('run05 restoration requires region_focus=0 and crushed_hold=0')
    # The legacy objective already supervises radiance, chroma and gradients.
    # Its outside term compares to the target and remains. Only the separate
    # zero-residual prior conflicts with dequantisation corrections.
    return losses['total'] - 0.10 * losses['residual_outside']
