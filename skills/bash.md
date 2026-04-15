---
name: bash
domain: all
priority: high
when_to_use: Load when the planner should prefer shell or Python edits first
---

# Skill: Bash Execution

- Current planning mode: bash-first. Prefer `bash_execution` for the next decision unless direct GUI manipulation is obviously required right now.
- Once bash mode is selected, keep using `bash_execution` for subsequent decisions. Do not switch to `gui_action` unless the system has re-selected GUI mode.
- Use this skill when the task can be completed more reliably by editing files, transforming tabular data, or inspecting outputs from the shell.
- Prefer one concise command or one short Python snippet that completes a meaningful chunk of work.
- Keep commands non-interactive.
- The default user is `user`, so use `/home/user` instead of `~` when passing file paths into Python or other tools that may not expand the shell shorthand. If sudo is needed, the password is `password`.
- Before editing any document, spreadsheet, or data file from bash, first locate the exact file path you will modify. Do not assume filenames or invent placeholder paths.
- Before modifying a file from bash, create a backup copy first so you can recover or compare if the edit goes wrong.
- Once the target file path is known, edit the file directly with a Python script or another file-editing command. Do not rely on GUI-only state, fake macro files, or format-conversion commands as a substitute for editing the real file.
- After changing files or data via bash, verify the result explicitly. If GUI confirmation is needed later, switch back to GUI rather than forcing everything through bash.
- If bash has already failed on the current approach, do not keep retrying slight command variants. Reassess whether the GUI is the safer path.

### Bash examples

```json
{
    "thought": "I need the exact workbook path before editing so the Python script modifies the real file rather than a guessed location.",
    "subgoal": "Locate the target workbook",
    "tool": "bash_execution",
    "input": "find /home/user -type f \\( -name '*.ods' -o -name '*.xlsx' \\) 2>/dev/null | head -20"
}
```

```json
{
    "thought": "Now that I know the workbook path, I can update the real file directly and save text-form values without relying on GUI state.",
    "subgoal": "Update the workbook data",
    "tool": "bash_execution",
    "input": "python3 - <<'PY'\nfrom openpyxl import load_workbook\npath = '/home/user/example.xlsx'\nwb = load_workbook(path)\nws = wb.active\ncell = ws['A2']\ncell.value = 'text_value'\ncell.number_format = '@'\nwb.save(path)\nprint(path)\nPY"
}
```
