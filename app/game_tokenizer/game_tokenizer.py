from tokenizers import Tokenizer


# class GameTokenizer():
#     def __init__(self):
#         super().__init__()


def _convert_token_to_id_with_added_voc(token, added_tokens_encoder):
    if token is None:
        return None

    if token in added_tokens_encoder:
        return added_tokens_encoder[token]

def create_func1(sep_token_id, cls_token_id):
    def get_special_tokens_mask(token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [sep_token_id, cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    return get_special_tokens_mask

def create_func2(added_tokens_encoder, mask_token_id):
    def convert_tokens_to_ids(tokens):
        """ Converts a single token, or a sequence of tokens, (str) in a single integer id
            (resp. a sequence of ids), using the vocabulary.
        """
        if tokens == '[MASK]':
            return mask_token_id

        if tokens is None:
            return None

        if isinstance(tokens, str):
            return _convert_token_to_id_with_added_voc(tokens, added_tokens_encoder)

        ids = []
        for token in tokens:
            ids.append(_convert_token_to_id_with_added_voc(token, added_tokens_encoder))
        return ids

    return convert_tokens_to_ids




