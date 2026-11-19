# Rapeseed in Glass — 3D Study 01

一束高精度油菜花插在透明直筒玻璃花瓶中的 PBR 3D 资产，按参考照片的花束姿态重建。参考图只用于构图与姿态，未作为贴图使用。

![Hero](renders/rapeseed_vase_hero_1440x1800.png)

## 交付文件

| 文件 | 说明 |
| --- | --- |
| `model/rapeseed_vase_high.glb` | **主资产**。PBR、透射玻璃、透明水面、独立对象节点。12.3 MB |
| `model/rapeseed_vase_high.obj` + `.mtl` | 高精度四边形为主的静态几何；适合 DCC / 线框导入。MTL 不含折射扩展。34.2 MB |
| `model/rapeseed_vase_wireframe.glb` | 全息线框 LOD，107,608 三角面，保留全部分离节点。3.6 MB |
| `model/rapeseed_vase_wireframe.obj` + `.mtl` | 线框 LOD 的 OBJ 版本。8.5 MB |
| `model/rapeseed_vase_manifest.json` | 尺寸、统计、材质、节点结构、文件清单。 |
| `index.html` + `viewer/` | 本地实时预览页，带自动旋转、线框模式、GLB/OBJ 下载。 |
| `renders/` | 白底主视觉、花朵细节、水面细节、背面与线框渲染。 |
| `scripts/` | 可复现的程序化建模与导出脚本。 |
| `source/reference.jpg` | 用户提供的姿态参考。 |

## 规格

- 单位：米。`1 unit = 1 m`，Y 轴向上。
- 原点：花瓶底面中心，底面位于 `y = 0`。
- 外形框：`0.251 × 0.439 × 0.147 m`（宽 × 高 × 深）。
- 花瓶：直筒，高 `0.176 m`，外径 `0.072 m`，壁厚约 `3.55 mm`；水面 `0.1082 m`。
- 结构：**289 个独立对象 / 节点**
  - 102 朵四瓣油菜花，每朵是独立节点；花瓣、花心、花萼是节点内分别命名的独立几何；
  - 165 根独立花茎节点（主茎、分叉、花梗、花苞梗分别命名）；
  - 18 个花苞；
  - 玻璃花瓶、水体积、水面、水线 4 个独立对象。
- 高精度版：**432,536 三角面**；线框版：**107,608 三角面**。
- 所有对象都是闭合流形几何（631 个 watertight parts）；无松散边、无三角面退化；花瓣使用规则参数网格，花茎使用环状四边面管体，花瓶为 128 段车削闭合玻璃壳。
- OBJ 保留四边形面片，GLB 中按规范三角化。

## 姿态与造型

- 中心一团密集花簇，左下有较低的花簇，右侧稍松，上方有高的主枝；
- 两根较高的单花枝（左上、右上）拉开轮廓；
- 花苞穿插在花枝之间，花茎自然分叉、弯曲；
- 参考图中没有明显的大叶片，因此模型刻意保持“只有花与细茎”的干净轮廓；
- 花为四片宽大的明黄色花瓣与小型金色花心，符合油菜花特征。

## PBR 材质

- 花瓣：4 个哑光明黄变体，roughness ≈ 0.76–0.80，metallic = 0。
- 花心/花药：暖金色高粗糙度材质。
- 花茎：嫩绿/中绿/深绿三档，roughness ≈ 0.58–0.66。
- 玻璃：`KHR_materials_transmission`、`KHR_materials_ior`、`KHR_materials_volume`、`KHR_materials_clearcoat`。
- 水体积：独立透射介质，IOR 1.333；水面与水线使用 alpha 混合加 clearcoat，保证在实时预览和白色背景上仍然可读。
- 所有网格带 UV（花瓣、叶片、花瓶、水面均有展开），可直接追加程序化纹理。
- **OBJ/MTL 无法表达 glTF 的折射扩展**；要查看玻璃/水，请使用 GLB 和支持 transmission 的查看器（本项目的实时页就是）。

## 打开实时预览

```bash
cd "rapeseed-vase"
python3 -m http.server 4175 --bind 127.0.0.1
# 打开 http://127.0.0.1:4175/
```

或者双击 **`start.command`**。

页面参数：

- 默认：带界面，自动缓慢旋转；
- `?still=1` 停止自动旋转；
- `?a=-6` 指定初始角度（度）；
- `?wire=1` 直接进入线框模式；
- `?clean=1` 隐藏界面，用于截图；
- `?view=detail` / `?view=vase` 聚焦花朵或水面；
- `?white=1` 使用纯白背景（默认是极浅的暖白影棚渐变）。

## 渲染预览

| 图 | 文件 |
| --- | --- |
| 白底主视觉 | `renders/rapeseed_vase_hero_1440x1800.png` |
| 花朵细节 | `renders/rapeseed_vase_flowers_1400.png` |
| 花瓶与水面 | `renders/rapeseed_vase_water_1280x1400.png` |
| 线框 | `renders/rapeseed_vase_wireframe_1440x1800.png` |
| 背面 | `renders/rapeseed_vase_back_1280x1600.png` |
| 预览界面 | `renders/viewer_interface_1440x900.png` |

## 关于玻璃显示

物理透射玻璃在纯白背景上本来就很难读出体积。这个资产同时提供了：
1. GLB 内准确闭合的玻璃壳与透射/体积材质；
2. 独立的水面 alpha 层与水线，使玻璃内的水位在实时预览中清楚可见；
3. 预览页使用极浅的暖白影棚渐变和边缘反射来展示玻璃。

在离线渲染器（Blender/Cycles、V-Ray 等）里，可以直接使用玻璃和水体积材质；在实时查看器中，建议开启 transmission、环境反射和适当的背景对比。


## 微光效果

在原白底展示上加入了一层很轻的金色柔光：

- 每朵油菜花瓣带有极淡的暖金色自发光，并随呼吸轻微变化；
- 花簇后方有淡金色光晕；
- 周围漂浮 380 颗金色花粉微粒，带柔和闪烁；
- 保留了原白底、绿茎、玻璃花瓶和水面，不改变花束造型。

光晕与花粉属于实时页面效果，GLB 本身仍是干净的 PBR 花束模型。
