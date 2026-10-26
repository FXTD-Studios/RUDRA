"""Versioned source-code evidence for missing highlight range.

Legacy models reconstruct near-white detail from code .94 upward. The
observed-clip policy reserves highlight invention for the last 8-bit bin;
code254 and below remain in the source-preserving branch. This is an input
observability rule, not proof that a bright pixel represents a light source.
"""

POLICIES = ('legacy_nearwhite', 'observed_clip_v1')
DEFAULT_POLICY = 'legacy_nearwhite'


def reconstruction_bands(policy=DEFAULT_POLICY):
    if policy == 'legacy_nearwhite':
        return (.94, .99), (1 / 255, 5 / 255)
    if policy == 'observed_clip_v1':
        return (254 / 255, 1.), (1 / 255, 5 / 255)
    raise ValueError(f'unknown reconstruction policy: {policy!r}')
