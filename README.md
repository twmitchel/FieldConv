# FieldConv
Field Convolutions for CNNs on Surfaces

Much of this code is adapted from the publicly available code for HSN (Wiersma et al. 2020) [https://github.com/rubenwiersma/hsn], which provides an excellent template for implementing equivariant surface networks. 

 
## Dependencies

PyTorch (V >= 1.70) [https://pytorch.org/]

PyTorch Geometric + Dependencies [https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html]

Progressbar2 [https://pypi.org/project/progressbar2/]

Suitesparse [https://people.engr.tamu.edu/davis/suitesparse.html]

Vector Heat extension:

Exposes select methods from Geometry Central [https://geometry-central.net/] to facilitate logarithm + transport computations. Install with

pip install ./vectorheat


## Concepts

Specific concepts discussed in the paper and supplement are implemented in the following modules:

Field Convolution (Section 4, Equation (7) )
./nn/field_conv.py

Field Convolution ResNetBlocks (Section 5, Figure 2)
./nn/fc_resnet_block

Computation of pointwise ECHO descriptors (Section 5, Mitchel et al. 2020)
./nn/echo_des.py 

ECHO Blocks (Section 5)
./nn/echo_block.py

Complex Linearities (Section 5)
./nn/tangent_lin.py

Complex Non-linearities (Section 5, Equation (8) )
./nn/tangent_nonlin.py

Learnable "gradient" (Supplement A)
./nn/trans_field.py

Inital layer taking scalar features to tangent vector features (Section 5, Supplement A)
./nn/lift_block.py

Precomputation: filter support (Section 6)
./transforms/support_graph.py

Precomputation: Computation of logarithm maps and transport using the Vector Heat Method  (Section 6.1, Sharp et al. 2019)
./transforms/vector_heat.py

Precomputation: Organization of computed quanities for input to network (Section 6.1)
./transforms/fc_precomp.py

Utilities: Interpolation weights, complex arithmetic, descriptor binning, vector magnitude and normalization, etc
./utils/field.py








