"""Minimal utility helpers needed by STD reproduction.

This is a stripped-down copy — only the functions actually called by
src/std_repro/ are kept so we don't pull in unused SpecVLM dependencies.
"""


def get_last_video_idx(input_ids, video_token_id):
    """Return the last position of *video_token_id* in *input_ids*, or -1."""
    last_video_idx = -1
    for i in range(len(input_ids) - 1, -1, -1):
        if input_ids[i] == video_token_id:
            last_video_idx = i
            break
    return last_video_idx
