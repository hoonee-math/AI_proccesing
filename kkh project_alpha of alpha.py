from tkinter import *

def create_image_menu():
    global menu_frame
    for widget in menu_frame.winfo_children():
        widget.destroy()

    menu_data = {
        "분류 전체보기": {
            "설치": 1,
            "머신러닝 (지도학습)": {
                "k최근접 이웃": 4,
                "회귀분석": 2,
                "결정트리": 0,
                "로지스틱 회귀분석": 1
            },
            "머신러닝 (비지도학습)": 0,
            "스케일 조정": 0,
            "k 평균 군집": 0,
            "딥러닝": 2,
            "분류": 2,
            "차원축소": 1,
            "PCA (주성분분석)": 1
        }
    }

    def create_submenu(parent, data, indent=0):
        for key, value in data.items():
            if isinstance(value, dict):
                count = sum(value.values()) if all(isinstance(v, int) for v in value.values()) else sum(v if isinstance(v, int) else sum(v.values()) for v in value.values())
                label = Label(parent, text="  " * indent + f"{key} ({count})", anchor="w", justify=LEFT)
                label.pack(fill=X, padx=10, pady=2)
                create_submenu(parent, value, indent + 1)
            else:
                label = Label(parent, text="  " * indent + f"{key} ({value})", anchor="w", justify=LEFT)
                label.pack(fill=X, padx=10, pady=2)

    total_count = sum(value if isinstance(value, int) else sum(value.values()) for value in menu_data["분류 전체보기"].values())
    Label(menu_frame, text=f"분류 전체보기 ({total_count})", font=("Arial", 12, "bold"), anchor="w", justify=LEFT).pack(fill=X, padx=10, pady=5)
    create_submenu(menu_frame, menu_data["분류 전체보기"])

# 메인 윈도우 생성
root = Tk()
root.title("메뉴 예시")

# 메뉴 프레임 생성
menu_frame = Frame(root)
menu_frame.pack(side=LEFT, fill=Y)

create_image_menu()

root.mainloop()