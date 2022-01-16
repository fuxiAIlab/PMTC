import os

top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
static_dir = os.path.join(top_dir, 'static')
BPE_TOKENIZER_PATH = os.path.join(static_dir, 'bpe_new.str')
BPE_GAME_ID_CN_CHAR_MAP_PATH = os.path.join(static_dir, 'game_id_cn_char.dict')
