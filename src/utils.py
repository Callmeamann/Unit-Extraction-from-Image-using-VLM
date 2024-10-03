import re
from src.constants import entity_unit_map

# Extract value and unit from text using regex
def extract_value_and_unit(text, entity_name):
    allowed_units = entity_unit_map.get(entity_name, [])
    pattern = re.compile(r"(\d+(\.\d+)?)\s*(" + "|".join(allowed_units) + ")")
    match = pattern.search(text)
    if match:
        return f"{match.group(1)} {match.group(3)}"
    return ""

class CFG:
    ### generation
    max_new_tokens = 200
    
    ### hardware
    device = 'cuda' # 'cuda', 'cpu'