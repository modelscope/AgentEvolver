def read_markdown_and_extract_sections(
    markdown_text,
    expected_sections=["current step", "previous instruction code", "relevant environment feedback", "next-step instruction code"],
    default_placeholder="❌ not available."
):
    sections = {}
    # if not isinstance(markdown_text, str):
    #     markdown_text = markdown_text.content_for_future
    lines = markdown_text.splitlines()
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("# "):
            section_name = line[2:].strip().lower()
            current_section = section_name
            sections[current_section] = []
        elif current_section:
            sections[current_section].append(line)

    for key in list(sections.keys()):
        if key not in expected_sections:
            sections.pop(key, None)

    section_to_return = {k: "\n".join(v) for k, v in sections.items()}
    find_all_expected_sections = True
    find_no_expected_sections = True
    for section in expected_sections:
        if section not in section_to_return:
            section_to_return[section] = default_placeholder
            find_all_expected_sections = False
        else:
            find_no_expected_sections = False


    return section_to_return, find_all_expected_sections, find_no_expected_sections
