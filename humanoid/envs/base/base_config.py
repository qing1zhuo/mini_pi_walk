"""配置类的基础工具。

项目使用“嵌套类”给配置分组，例如：

    class RobotCfg(BaseConfig):
        class env:
            num_envs = 4096

        class commands:
            class ranges:
                lin_vel_x = [-1.0, 1.0]

创建 ``cfg = RobotCfg()`` 时，BaseConfig 会递归实例化这些嵌套类。
之后可以通过 ``cfg.env.num_envs``、``cfg.commands.ranges.lin_vel_x`` 读取配置，
也可以用 ``cfg.env.num_envs = 8`` 为当前配置对象覆盖某个默认值。

注意：嵌套类实例彼此独立，但类中定义的 list/dict 等可变值没有被深拷贝。
如需修改列表，优先整体赋一个新列表，避免用 append 等原地操作影响其他配置实例。
"""

import inspect  # Python 标准库：检查一个对象是类、函数还是其他类型。


class BaseConfig:
    """所有分层配置类的基类。

    子类只需声明配置分组和默认值，不需要自己实现初始化逻辑。
    """

    def __init__(self) -> None:
        """创建配置对象，并递归实例化其中所有嵌套配置类。

        例如，实例化 ``RobotCfg`` 后，类属性 ``RobotCfg.env`` 对应的类会被
        转换为当前对象自己的 ``cfg.env`` 实例。
        """
        self.init_member_classes(self)

    @staticmethod
    def init_member_classes(obj):
        """递归地把 ``obj`` 中的嵌套类替换为该类的实例。

        ``@staticmethod`` 表示这个工具函数不依赖某个 BaseConfig 实例，
        因而没有 ``self`` 参数。递归处理 env、commands、commands.ranges 等
        任意层级时，只需把当前正在处理的对象作为 ``obj`` 传入。
        """
        # dir(obj) 返回 obj 可访问的全部属性名，包括继承来的配置分组。
        for key in dir(obj):
            # obj.__class__ 是 obj 自身所属的类。如果不跳过它，下面会再次
            # 实例化同一种配置并无限递归。
            if key == "__class__":
                continue

            # getattr(obj, key) 等价于访问 obj.<key>，但 key 可以在循环中变化。
            var = getattr(obj, key)

            # 只处理“值本身是一个类”的属性；数字、字符串、列表和方法会跳过。
            if inspect.isclass(var):
                # var() 调用该嵌套类的构造函数，创建一个配置分组实例。
                instance = var()

                # setattr(obj, key, instance) 等价于 obj.<key> = instance。
                # 这会让当前配置对象持有自己的分组实例，而不是继续共用类对象。
                setattr(obj, key, instance)

                # 当前分组内部还可能包含嵌套类，例如 commands.ranges，
                # 因此继续递归，直到所有层级都被实例化。
                BaseConfig.init_member_classes(instance)
