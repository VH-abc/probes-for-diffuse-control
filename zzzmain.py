import sweep
import config

'''
sweeps:
config.DEFAULT_LAYER_SWEEP at position letter
["last", "first", "middle", "letter", "all_appended"] at config.DEFAULT_LAYER
'''

sweep.sweep_layers(
    layers=config.DEFAULT_LAYER_SWEEP,
    token_position=config.DEFAULT_TOKEN_POSITION,
    num_examples=config.DEFAULT_NUM_EXAMPLES,
    skip_cache=False,
    skip_analysis=False,
    filter_reliable=True,
    reliable_questions_file="experiments/gemma-3-12b/reliable_questions.json"
)

sweep.sweep_positions(
    layer=config.DEFAULT_LAYER,
    positions=config.DEFAULT_POSITION_SWEEP,
    num_examples=config.DEFAULT_NUM_EXAMPLES,
    skip_cache=False,
    skip_analysis=False,
    filter_reliable=True,
    reliable_questions_file="experiments/gemma-3-12b/reliable_questions.json"
)