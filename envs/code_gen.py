import gym


class CodeGen(gym.Env):
    def __init__(self):
        self.code = None

    def reset(self):
        self.code = ""

    def step(self, action):
        if action == 0:
            try:
                code = self.code + "\nf(2, 3)"
                result = eval(self.code)

                if result == self.expected:
                    return self.code, 1, True, None
                else:
                    return self.code, -1, True, None
            except SyntaxError:
                return self.code, -10, True, None
            except:
                return self.code, -5, True, None
        else:
            char = chr(action)
            self.code += char
            return self.code, 0, False, None
