from typing import List
from constants import MAX_CHUNK_LENGTH

def split_text_into_chunks(text: str, max_length: int = MAX_CHUNK_LENGTH) -> List[str]:
    chunks = []
    current_pos = 0
    while current_pos < len(text):
        end_pos = min(current_pos + max_length, len(text))
        split_index = -1
        for sep in ['.', '?', '!']:
            found_index = text.rfind(sep, current_pos, end_pos)
            if found_index != -1:
                split_index = max(split_index, found_index)

        if split_index != -1 and end_pos < len(text):
            chunk = text[current_pos : split_index + 1]
            current_pos = split_index + 1
        elif end_pos < len(text):
            space_index = text.rfind(' ', current_pos, end_pos)
            if space_index != -1 and space_index > current_pos:
                chunk = text[current_pos : space_index]
                current_pos = space_index + 1
            else:
                chunk = text[current_pos : end_pos]
                current_pos = end_pos
        else:
            chunk = text[current_pos:]
            current_pos = len(text)

        cleaned_chunk = chunk.strip()
        if cleaned_chunk:
            chunks.append(cleaned_chunk)
    return chunks