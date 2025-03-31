def extract_j_content_string(filepath, output_filepath='data.txt'):
    """
    使用字符串查找方法从文件中提取每行 'j:{...}' 花括号内的内容，并保存到输出文件。

    Args:
        filepath (str): 包含数据的文件路径。
        output_filepath (str): 输出文件路径，默认为'data.txt'。

    Returns:
        list: 一个包含所有提取到的字符串内容的列表。
              如果某行格式不符，则该行会被跳过。
    """
    extracted_contents = []
    start_marker = "j:{"
    end_marker = "}"

    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                try:
                    # 查找 'j:{' 的起始位置
                    start_index = line.index(start_marker)
                    # 从 'j:{' 之后开始查找第一个 '}' 的位置
                    end_index = line.index(end_marker, start_index + len(start_marker))

                    # 提取两个标记之间的内容
                    content = line[start_index + len(start_marker) : end_index]
                    extracted_contents.append(content.strip())

                except ValueError:
                    # 如果 .index() 找不到标记，会抛出 ValueError
                    # 跳过这一行，或者打印警告
                    if line.strip(): # 避免对空行发出警告
                         print(f"警告：在第 {i+1} 行未找到 '{start_marker}' 或 '{end_marker}'。")
    except FileNotFoundError:
        print(f"错误：文件未找到 '{filepath}'")
        return None

    # 将提取的内容保存到输出文件
    try:
        with open(output_filepath, 'w') as f:
            for content in extracted_contents:
                f.write(content + '\n')
        print(f"提取的数据已保存到 {output_filepath}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

    return extracted_contents

# --- 使用示例 ---
filepath = 'calib_joint_point.txt'
contents_str = extract_j_content_string(filepath)

if contents_str is not None:
    print("\n使用字符串查找提取的内容:")
    for i, item in enumerate(contents_str, 1):
        print(f"j{i}: {item}")