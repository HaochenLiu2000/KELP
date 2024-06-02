import json

def clean_json_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    cleaned_lines = []
    current_json = ""

    for line in lines:
        line = line.strip()

        if line.startswith('{') and current_json == "":
            current_json = line
        elif line.endswith('}') and current_json != "":
            current_json += line
            cleaned_lines.append(current_json)
            current_json = ""
        elif current_json != "":
            current_json += line
        else:
            cleaned_lines.append(line)

    return cleaned_lines

def write_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        for obj in data:
            json_line = json.dumps(obj, ensure_ascii=False)
            file.write(json_line + '\n')

input_file_path = 'one-hop.txt' 
output_file_path = 'one-hop.jsonl' 

cleaned_lines = clean_json_lines(input_file_path)


with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write('\n'.join(cleaned_lines))


