# Vulkan Coordinate Transformation Helper
## Overview
This repository contains a Python script designed to demonstrate and handle the various steps involved in the process of transforming coordinates in a Vulkan-based graphics environment. The script showcases how to correctly derive and apply Projection (P), Coordinate System Change (X), View (V), and Model (M) matrices, collectively referred to as PXVM, within the context of 3D graphics rendering in Vulkan. It's intended as a comprehensive guide and toolkit for understanding and implementing the necessary transformations to correctly render 3D objects in Vulkan.

I made it when I was struggling to get the math working for my project, [Velecs Engine C++](https://www.matthewgreen.gg/velecs-and-harvest-havoc/). After many many hours of debugging and much frustration, I discovered an oversight in my learning of Vulkan; there is a coordinate system change between world and clip space. The result of not reading things closely. In my struggle I made this script to quickly test changes in the calculations of rendering matrices all in a small single Python script rather than many C++ files scattered around.

## Key Features
- __Matrix Derivation:__ Step-by-step derivation of Projection, View, and Model matrices.
- __Coordinate System Conversion:__ Detailed handling of the transition between Vulkan's world space to clip space and finally to normalized device coordinate (NDC) space.
- __Examples and Tests__: Includes actual test examples demonstrating the practical application of these matrices and transformations.

## Special Emphasis
An essential aspect to note, and a focal point of this script, is that the coordinate system changes between Vulkan's world space and clip space. Vulkan's world space is a right-handed coordinate system where the Z-axis points out of the screen (towards the viewer) and the X-axis points up. In contrast, Vulkan's clip, NDC, and Framebuffer spaces are also a right-handed system but is different in that the Z-axis points into the screen (away from the viewer) and the Y-axis points down. This difference is crucial and can easily lead to confusion and mistakes. This script aims to clarify these differences and demonstrate the correct approach to handling them.
<div align="center">
    <figure>
        <img src="docs/README.md/view-space-prep-for-proj-fade.gif" alt="coordinate system change">
        <figcaption>Image Source: <a href="https://johannesugb.github.io/gpu-programming/setting-up-a-proper-vulkan-projection-matrix/#prepare-for-perspective-projection">Johannes' Guide on Setting Up a Proper Vulkan Projection Matrix</a></figcaption>
    </figure>
</div>


## Primary Resource
A significant portion of the knowledge and methodologies applied in this script are derived from Johannes' detailed guide on setting up a proper Vulkan projection matrix. This resource provided invaluable insights and served as a primary guide in understanding and implementing the correct transformations. You can find the guide here: [Setting Up a Proper Vulkan Projection Matrix](https://johannesugb.github.io/gpu-programming/setting-up-a-proper-vulkan-projection-matrix/). It is highly recommended for anyone working with or learning about Vulkan's coordinate systems and transformations.

## Additional References
While the primary concepts and implementations are based on Johannes' guide, several other resources have contributed to the understanding and context of this subject:

- __[Depth Precision Visualized](https://developer.nvidia.com/content/depth-precision-visualized)__: Nvidia's exploration of depth precision in graphics.
- __[Vulkan Perspective Matrix](https://vincent-p.github.io/posts/vulkan_perspective_matrix/)__: A detailed look at perspective matrix specifics in Vulkan.
- __[Understanding the View Matrix](https://www.3dgep.com/understanding-the-view-matrix/)__: Comprehensive guide on view matrix concepts and calculations.
Usage
