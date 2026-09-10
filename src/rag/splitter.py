from typing import List

def split_into_chunks(content: str, is_path: bool = True) -> List[str]:
    if is_path:
        with open(content, 'r', encoding='utf-8') as file:
            text = file.read()
    else:
        text = content
    return [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]