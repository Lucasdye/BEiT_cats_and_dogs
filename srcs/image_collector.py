from PIL import Image 

def load_image_collection(path: str, max: int, format: str) -> list:
    """
    Summary:
    This methods retrieves a collection of image which the path
    ends with "_" from which is appended the 0 to max numbers and the format
    format to retrieve all collection into a list e.g path = "cat_" nb = 3
    will load path/cat_0 path/cat_1 path/cat_3.
    """
    collection = []
    for i in range(max):
        new_path = path[:path.rfind("_") + 1]
        new_path += str(i + 1) + format
        print(new_path)
        new_image = Image.open(new_path)
        collection.append(new_image)
    return collection


def RGB_convert(lst: list[Image]) -> list:
    """
    Summary:
    Converts list of images into RGB format and returns it.
    """
    new_lst = []
    for i in range(len(lst)):
        new_image = lst[i].convert("RGB")
        new_lst.append(new_image)
    return new_lst


# class image_collector:
#     """
#     This class is intended to facilitate 
#     """
#     def __init__():
#     def __str__(self):
#         return f"{self.collection}"
#     image_collector.load_image_collection = load_image_list
#     image_collector.RGB_convert = RGB_convert
