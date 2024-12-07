import sys,os

def find_opengait(path):
    while True:
        des_path = os.path.join(path, 'opengait')
        if os.path.exists(des_path):
            break
        else:
            path = os.path.dirname(path)

    sys.path.append(path)
    sys.path.append(des_path)
    
    for module in os.listdir(des_path):
        module_path = os.path.join(des_path, module)
        if os.path.isdir(module_path):
            sys.path.append(module_path)

find_opengait(sys.path[0])