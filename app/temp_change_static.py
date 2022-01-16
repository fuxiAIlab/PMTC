import os
import json
import ipdb


def load_save_json(json_path, mode, verbose=1, encoding='utf-8', data=None):
    if mode == 'save':
        assert data is not None
        with open(json_path, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            if verbose >= 1:
                print(f"save json data to {json_path}")
    elif mode == 'load':
        if os.path.isfile(json_path):
            with open(json_path, 'r', encoding=encoding) as f:
                response = json.load(f)
            if verbose >= 1:
                print(f"load json from {json_path} success")
        else:
            raise Exception(f"{json_path} does not exist!")
        return response
    else:
        raise NotImplementedError


def main():
    path = '../static/game_id_cn_char.dict'
    data = load_save_json(path, 'load')
    new_data = {'[UNK]': '嗯', '0': '一'}
    load_save_json(path, 'save', data=new_data)


if __name__ == '__main__':
    main()
