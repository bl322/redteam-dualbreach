"""DualBreach red-team evaluation package.

精简包：只包含 DualBreach（TDI → 代理围栏 → MTO）及其评测所需的依赖闭包。
不含 DARWIN / CoT / CC-BoS 攻击实现与 Gradio 界面。
"""

__all__ = ["dbservice", "dualbreach"]
