from typing import Optional

BLOCKS = {
    'content': ['unet.up_blocks.0.attentions.0'],
    'style': ['unet.up_blocks.0.attentions.1'],
}


def is_belong_to_blocks(key, blocks):
    try:
        for g in blocks:
            if g in key:
                return True
        return False
    except Exception as e:
        raise type(e)(f'failed to is_belong_to_block, due to: {e}')


def filter_lora(state_dict, blocks_):
    try:
        return {k: v for k, v in state_dict.items() if is_belong_to_blocks(k, blocks_)}
    except Exception as e:
        raise type(e)(f'failed to filter_lora, due to: {e}')


def scale_lora(state_dict, alpha):
    try:
        return {k: v * alpha for k, v in state_dict.items()}
    except Exception as e:
        raise type(e)(f'failed to scale_lora, due to: {e}')


def close_block(state_dict, block_index, prefix):
    """Remove keys belonging to the specified transformer block from state_dict.
    prefix should be the block path before transformer_blocks (e.g. 'unet.up_blocks.0.attentions.1').
    block_index is the transformer block index (0-9).
    """
    pattern = f'{prefix}.transformer_blocks.{block_index}.'
    return {k: v for k, v in state_dict.items() if pattern not in k}


def close_blocks(state_dict, block_indices, prefix):
    """Remove keys belonging to any of the specified transformer blocks from state_dict.
    block_indices is an iterable of block indices (0-9).
    """
    result = state_dict
    for idx in block_indices:
        result = close_block(result, idx, prefix)
    return result


def get_target_modules(unet, blocks=None):
    try:
        if not blocks:
            blocks = [('.').join(blk.split('.')[1:]) for blk in BLOCKS['content'] + BLOCKS['style']]

        attns = [attn_processor_name.rsplit('.', 1)[0] for attn_processor_name, _ in unet.attn_processors.items() if
                 is_belong_to_blocks(attn_processor_name, blocks)]

        target_modules = [f'{attn}.{mat}' for mat in ["to_k", "to_q", "to_v", "to_out.0"] for attn in attns]
        return target_modules
    except Exception as e:
        raise type(e)(f'failed to get_target_modules, due to: {e}')


