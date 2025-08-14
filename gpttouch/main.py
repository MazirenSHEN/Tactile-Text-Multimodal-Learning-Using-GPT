# touch_qa_rag.py
from TactileQASystem import TouchQAModel
from dotenv import load_dotenv

load_dotenv()
if __name__ == "__main__":
    model = TouchQAModel()  # 初始化

    history = []  # 多轮对话历史
    image_path = "data/"

    # 第一轮
    user_question = "Please describe the tactile attributes of the objects in the image."
    reply = model.answer(image_path, user_question, history)
    print("Answer1：", reply)
    history.append({"question": user_question, "answer": reply})

    # # 第二轮（举例，可选）
    # user_question2 = "What's the difference in touch between it and the common wooden surface?"
    # reply2 = model.answer(image_path, user_question2, history)
    # print("Answer2：", reply2)
    # history.append({"question": user_question2, "answer": reply2})
