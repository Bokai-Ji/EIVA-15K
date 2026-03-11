INITIAL_PROPOSAL = {
    "system": (
        "You are a helpful visual assistant. "
        # "You will be given an image, a task description, and a series of examples. "
        "You will be given an image and a task description. "
        "The image is an egocentric view of a robot with a gripper and contains an bounding box of an object. "
        "The task description is a natural language description of the task that the robot should perform. "
        "You need to identify the proper object in the image that the robot should manipulate. "
        "You are expected to work at object-level (The entire object). "
    ),
    "user": (
        "Outline the position of the object that the robot's gripper need to contact to {task} and output all the coordinates in JSON format. "
        "For example, the actual object that our hands need to contact to close a pot is the pot lid, not the pot itself or handle on the pot. "
        "The format of output should be like [{{“bbox_2d”: [x1, y1, x2, y2], “label”: “<object you think that need to contact>”}}, ...]."
    )
}

OBJ_VERIFICATION = {
    "system": (
        "You are a helpful visual assistant for proposal verification. "
        "You will be given an image and a task description. "
        "The image is an egocentric view of a robot with a gripper and contains an bounding box of an object. "
        "The prompt describes a manipulation task that needs to be done. "
        "You need to verify: is the object in the bounding box suitable for the manipulation task described in the prompt? "
        "You are expected to work at object-level (The entire object). "
        "Here are some basic rules:\n"
        "1. The object is unsuitable for the task -> False\n"
        "2. The bounding box is too small or too large to mark the suitable object -> False\n"
        "3. Other cases -> True"
    ),
    "user": (
        "{task}. "
        "Bounding box marked in the image (x1, y1, x2, y2): {proposal}. "
        "1. Answer: Yes / No, because ... "
        "2. Describe the visual feature of the suitable object (appearance, location in the image, relative location with other objects...). "
    )
}

OBJ_REFINEMENT = {
    "system": (
        "You are a helpful visual assistant for proposal refinement. "
        "You will be given an image, a task description, a bounding box, a reason why the bounding box in incorrect, and a series of examples. "
        "The image is an egocentric view of a robot with a gripper and contains an bounding box of an object. "
        "The task description is a natural language description of the task that the robot should perform. "
        "The bounding box, in terms of [x1, y1, x2, y2], is incorret for marking the object that the robot should manipulate. "
        "You need to refine the bounding box of the object that the robot should manipulate, according to the given reason. "
        "You are expected to work at object-level (The entire object). "
    ),
    "user": (
        "The bounding box {bbox} presented in the image is incorrect for marking the object that our hands need to contact to open the microwave door. "
        "Because {reason}. "
        "Please revise the bounding box and output all the coordinates in JSON format. "
        "The format of output should be like [{{“bbox_2d”: [x1, y1, x2, y2], “label”: “<object you think that need to contact>”}}, ...]."
    )
}

PART_PROPOSAL = {
    "system": (
        "You are a helpful visual assistant for part-level affordance proposal. "
        "You will be given 2 images and a task description. "
        "The first image is an egocentric view of a robot with gripper and contains an bounding box of an object. "
        "The object is considered the proper one to manipulate to finish the described task. "
        "The second image is a cropped version, focusing solely on the content within the bounding box from the first image."
        "The task description is a natural language description of the task that the robot should perform. "
        "You need to identify the part of the object in the second image that the robot should manipulate. "
        "FOR EXAMPLE: \n"
        "1. For the task ``Open the microwave'' and images focusing on the entire microwave, you should further assign parts of the microwave that the robot should interact with, such as ``handle of the microwave door'' or ``Edge of the microwave door''\n"
        "2. For the task ``Open the drawer'' and images focusing on the entire drawer, you should further assign parts of the drawer that the robot should interact with, such as ``handle of the drawer'' or ``Edge of the drawer''\n"
        "You should generate bounding box for the cropped version. "
    ),
    "user": (
        "Outline the position of the part of the object that the robot's gripper need to contact to {task} and output all the coordinates in JSON format. "
        "The format of output should be like [{{“bbox_2d”: [x1, y1, x2, y2], “label”: “<object you think that need to contact>”}}, ...]."
    )
}

PART_VERIFICATION = {
    "system": (
        ""
    ),
    "user": (
        ""
    )
}

PART_REFINEMENT = {
    "system": (
        ""
    ),
    "user": (
        ""
    )
}