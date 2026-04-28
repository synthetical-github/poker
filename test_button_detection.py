import cv2
from action_executor import ActionExecutor

ae = ActionExecutor()

screenshots = [
    ("auto_screenshots/1777377616_3DQH_QS5HQH.png", "Hand4 flop (trips Q - no buttons expected)"),
    ("auto_screenshots/1777377691_KD4C_PRE.png", "Hand7 preflop (buttons worked in session)"),
    ("auto_screenshots/1777377488_QS4C_PRE.png", "Hand3 preflop"),
    ("auto_screenshots/1777377564_3DQH_PRE.png", "Hand4 preflop"),
    ("auto_screenshots/1777377458_KCAH_TH4C5C.png", "Hand1 flop (KC AH)"),
]

for fname, label in screenshots:
    img = cv2.imread(fname)
    if img is None:
        print(f"{label}: LOAD FAILED")
        continue
    r = ae.read_action_state(img)
    print(f"{label}:")
    print(f"  panel_visible={r['panel_visible']}")
    print(f"  actions={r['available_actions']}")
    print(f"  buttons_confirmed={r['buttons_confirmed']}")
    print(f"  is_my_turn={r['is_my_turn']}")
    print(f"  call_amount={r['call_amount']} raise_to={r['raise_to_amount']}")
    print()
