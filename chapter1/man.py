class Man:
    def __init__(self, name):
        self.name = name
        # Python中可以像self.name这样，通过在self后添加属性名来生成
        # 或访问实例变量
        print("Initialized: " + self.name)

    def hello(self):
        print("Hello, I'm " + self.name + "!")

m = Man("David")
m.hello()