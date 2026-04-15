---
name: gui
domain: all
priority: high
when_to_use: Load when the planner should interact through the visible GUI
---

# Skill: GUI Interaction

- Current planning mode: GUI-first. Prefer `gui_action` for the next decision unless the current screenshot clearly shows that shell work is the more reliable path.
- Use this skill when the visible interface is the main source of truth or when shell-side edits are brittle.
- Prefer short, purposeful `gui_action` sequences over one-click micro-steps when the sequence is stable.
- You may output multiple `pyautogui` statements in one `input`. The executor will parse them one by one, ground each mouse-position action before executing it, and use a fresh screenshot before the next sub-action.
- For mouse-position actions, put the target description in Python `# comments` inside `input`.
- Every grounded mouse-position action should carry a local `# comment`. If one `gui_action` contains multiple grounded mouse-position actions, provide per-action comments rather than relying on `thought`.
- If a text field is clearly identifiable, combine focus and typing in one `gui_action`.
- Use `wait` only for real async delays; do not replace ordinary observation steps with blind waiting.
- If this GUI-first path becomes blocked, the system may switch modes on the next planning round.
- Before `termination`, do one explicit verification action that inspects the requested final state.

### GUI action schema

Use these exact pyautogui APIs in `input`:
- Single click: `pyautogui.click(x, y)`
- Double click: `pyautogui.doubleClick(x, y)`
- Right click: `pyautogui.rightClick(x, y)`
- Hover/move: `pyautogui.moveTo(x, y)`
- Drag: `pyautogui.moveTo(x1, y1) # move to the drag start\npyautogui.dragTo(x2, y2, duration=0.5, button='left') # drag to the target`
- Type text: `pyautogui.write('text')`
- Press key: `pyautogui.press('enter')`
- Hotkey: `pyautogui.hotkey('ctrl', 'c')`
- Scroll a specific region: `pyautogui.moveTo(x, y) # move to the scroll target\npyautogui.scroll(amount) # amount < 0 for down, amount > 0 for up, keep amount within [-10, 10]`

### GUI examples

```json
{
    "thought": "The address bar is visible and I can open the target website in one short gui action.",
    "subgoal": "Open the target website",
    "tool": "gui_action",
    "input": "pyautogui.click(102, 234) # click the browser address bar at the top of the window\npyautogui.write('https://example.com')\npyautogui.press('enter')"
}
```

```json
{
    "thought": "I need to select a range by dragging from the source cell to the destination cell.",
    "subgoal": "Select the target range",
    "tool": "gui_action",
    "input": "pyautogui.moveTo(233, 566) # move to the source cell of the range\npyautogui.dragTo(444, 780, duration=0.5, button='left') # drag to the destination cell of the range"
}
```
