▓щ
╝І
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
ђ
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resourceѕ
«
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
Ј
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeіьout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
э
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
░
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleіжУelement_dtype"
element_dtypetype"

shape_typetype:
2	
Ъ
TensorListReserve
element_shape"
shape_type
num_elements(
handleіжУelement_dtype"
element_dtypetype"

shape_typetype:
2	
ѕ
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ
ћ
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
ѕ"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8гл
Ю
"simple_rnn_21/simple_rnn_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"simple_rnn_21/simple_rnn_cell/bias
ќ
6simple_rnn_21/simple_rnn_cell/bias/Read/ReadVariableOpReadVariableOp"simple_rnn_21/simple_rnn_cell/bias*
_output_shapes	
:ђ*
dtype0
║
.simple_rnn_21/simple_rnn_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*?
shared_name0.simple_rnn_21/simple_rnn_cell/recurrent_kernel
│
Bsimple_rnn_21/simple_rnn_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp.simple_rnn_21/simple_rnn_cell/recurrent_kernel* 
_output_shapes
:
ђђ*
dtype0
Ц
$simple_rnn_21/simple_rnn_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*5
shared_name&$simple_rnn_21/simple_rnn_cell/kernel
ъ
8simple_rnn_21/simple_rnn_cell/kernel/Read/ReadVariableOpReadVariableOp$simple_rnn_21/simple_rnn_cell/kernel*
_output_shapes
:	ђ*
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:*
dtype0
{
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ* 
shared_namedense_14/kernel
t
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes
:	ђ*
dtype0
Ћ
serving_default_input_38Placeholder*4
_output_shapes"
 :                  *
dtype0*)
shape :                  
}
serving_default_input_53Placeholder*(
_output_shapes
:         ђ*
dtype0*
shape:         ђ
Ћ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_38serving_default_input_53$simple_rnn_21/simple_rnn_cell/kernel"simple_rnn_21/simple_rnn_cell/bias.simple_rnn_21/simple_rnn_cell/recurrent_kerneldense_14/kerneldense_14/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *H
_output_shapes6
4:                  :         ђ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *.
f)R'
%__inference_signature_wrapper_7846529

NoOpNoOp
џ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Н
value╦B╚ B┴
Ц
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
* 
ф
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec*
д
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
'
0
1
2
3
4*
'
0
1
2
3
4*
* 
░
 non_trainable_variables

!layers
"metrics
#layer_regularization_losses
$layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*

%trace_0
&trace_1* 

'trace_0
(trace_1* 
* 

)serving_default* 

0
1
2*

0
1
2*
* 
Ъ

*states
+non_trainable_variables

,layers
-metrics
.layer_regularization_losses
/layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
0trace_0
1trace_1
2trace_2
3trace_3* 
6
4trace_0
5trace_1
6trace_2
7trace_3* 
М
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
>_random_generator

kernel
recurrent_kernel
bias*
* 

0
1*

0
1*
* 
Њ
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Dtrace_0* 

Etrace_0* 
_Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_14/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$simple_rnn_21/simple_rnn_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE.simple_rnn_21/simple_rnn_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"simple_rnn_21/simple_rnn_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1
2*

0
1
2*
* 
Њ
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

Ktrace_0
Ltrace_1* 

Mtrace_0
Ntrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╗
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_14/kerneldense_14/bias$simple_rnn_21/simple_rnn_cell/kernel.simple_rnn_21/simple_rnn_cell/recurrent_kernel"simple_rnn_21/simple_rnn_cell/biasConst*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *)
f$R"
 __inference__traced_save_7847159
Х
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_14/kerneldense_14/bias$simple_rnn_21/simple_rnn_cell/kernel.simple_rnn_21/simple_rnn_cell/recurrent_kernel"simple_rnn_21/simple_rnn_cell/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *,
f'R%
#__inference__traced_restore_7847183Хю
Є.
╗
while_body_7846835
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0I
6while_simple_rnn_cell_matmul_readvariableop_resource_0:	ђF
7while_simple_rnn_cell_biasadd_readvariableop_resource_0:	ђL
8while_simple_rnn_cell_matmul_1_readvariableop_resource_0:
ђђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorG
4while_simple_rnn_cell_matmul_readvariableop_resource:	ђD
5while_simple_rnn_cell_biasadd_readvariableop_resource:	ђJ
6while_simple_rnn_cell_matmul_1_readvariableop_resource:
ђђѕб,while/simple_rnn_cell/BiasAdd/ReadVariableOpб+while/simple_rnn_cell/MatMul/ReadVariableOpб-while/simple_rnn_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Б
+while/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp6while_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0└
while/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:03while/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђА
,while/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp7while_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0╣
while/simple_rnn_cell/BiasAddBiasAdd&while/simple_rnn_cell/MatMul:product:04while/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђе
-while/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp8while_simple_rnn_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0Д
while/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_25while/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђД
while/simple_rnn_cell/addAddV2&while/simple_rnn_cell/BiasAdd:output:0(while/simple_rnn_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђt
while/simple_rnn_cell/TanhTanhwhile/simple_rnn_cell/add:z:0*
T0*(
_output_shapes
:         ђК
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/simple_rnn_cell/Tanh:y:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: |
while/Identity_4Identitywhile/simple_rnn_cell/Tanh:y:0^while/NoOp*
T0*(
_output_shapes
:         ђх

while/NoOpNoOp-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"p
5while_simple_rnn_cell_biasadd_readvariableop_resource7while_simple_rnn_cell_biasadd_readvariableop_resource_0"r
6while_simple_rnn_cell_matmul_1_readvariableop_resource8while_simple_rnn_cell_matmul_1_readvariableop_resource_0"n
4while_simple_rnn_cell_matmul_readvariableop_resource6while_simple_rnn_cell_matmul_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ђ: : : : : 2\
,while/simple_rnn_cell/BiasAdd/ReadVariableOp,while/simple_rnn_cell/BiasAdd/ReadVariableOp2Z
+while/simple_rnn_cell/MatMul/ReadVariableOp+while/simple_rnn_cell/MatMul/ReadVariableOp2^
-while/simple_rnn_cell/MatMul_1/ReadVariableOp-while/simple_rnn_cell/MatMul_1/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Љ.
┐
while_body_7846625
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0I
6while_simple_rnn_cell_matmul_readvariableop_resource_0:	ђF
7while_simple_rnn_cell_biasadd_readvariableop_resource_0:	ђL
8while_simple_rnn_cell_matmul_1_readvariableop_resource_0:
ђђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorG
4while_simple_rnn_cell_matmul_readvariableop_resource:	ђD
5while_simple_rnn_cell_biasadd_readvariableop_resource:	ђJ
6while_simple_rnn_cell_matmul_1_readvariableop_resource:
ђђѕб,while/simple_rnn_cell/BiasAdd/ReadVariableOpб+while/simple_rnn_cell/MatMul/ReadVariableOpб-while/simple_rnn_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Б
+while/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp6while_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0└
while/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:03while/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђА
,while/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp7while_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0╣
while/simple_rnn_cell/BiasAddBiasAdd&while/simple_rnn_cell/MatMul:product:04while/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђе
-while/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp8while_simple_rnn_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0Д
while/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_25while/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђД
while/simple_rnn_cell/addAddV2&while/simple_rnn_cell/BiasAdd:output:0(while/simple_rnn_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђt
while/simple_rnn_cell/TanhTanhwhile/simple_rnn_cell/add:z:0*
T0*(
_output_shapes
:         ђК
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/simple_rnn_cell/Tanh:y:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: |
while/Identity_4Identitywhile/simple_rnn_cell/Tanh:y:0^while/NoOp*
T0*(
_output_shapes
:         ђх

while/NoOpNoOp-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"p
5while_simple_rnn_cell_biasadd_readvariableop_resource7while_simple_rnn_cell_biasadd_readvariableop_resource_0"r
6while_simple_rnn_cell_matmul_1_readvariableop_resource8while_simple_rnn_cell_matmul_1_readvariableop_resource_0"n
4while_simple_rnn_cell_matmul_readvariableop_resource6while_simple_rnn_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ђ: : : : : 2\
,while/simple_rnn_cell/BiasAdd/ReadVariableOp,while/simple_rnn_cell/BiasAdd/ReadVariableOp2Z
+while/simple_rnn_cell/MatMul/ReadVariableOp+while/simple_rnn_cell/MatMul/ReadVariableOp2^
-while/simple_rnn_cell/MatMul_1/ReadVariableOp-while/simple_rnn_cell/MatMul_1/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
У
Г
while_cond_7846834
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_7846834___redundant_placeholder05
1while_while_cond_7846834___redundant_placeholder15
1while_while_cond_7846834___redundant_placeholder25
1while_while_cond_7846834___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ђ: :::::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
╝8
┌
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7847003

inputs
initial_state_0A
.simple_rnn_cell_matmul_readvariableop_resource:	ђ>
/simple_rnn_cell_biasadd_readvariableop_resource:	ђD
0simple_rnn_cell_matmul_1_readvariableop_resource:
ђђ
identity

identity_1ѕб&simple_rnn_cell/BiasAdd/ReadVariableOpб%simple_rnn_cell/MatMul/ReadVariableOpб'simple_rnn_cell/MatMul_1/ReadVariableOpбwhilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  P
ShapeShapetranspose:y:0*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЋ
%simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp.simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0ю
simple_rnn_cell/MatMulMatMulstrided_slice_1:output:0-simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЊ
&simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp/simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Д
simple_rnn_cell/BiasAddBiasAdd simple_rnn_cell/MatMul:product:0.simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђџ
'simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp0simple_rnn_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0Ќ
simple_rnn_cell/MatMul_1MatMulinitial_state_0/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
simple_rnn_cell/addAddV2 simple_rnn_cell/BiasAdd:output:0"simple_rnn_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђh
simple_rnn_cell/TanhTanhsimple_rnn_cell/add:z:0*
T0*(
_output_shapes
:         ђn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : И
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0initial_state_0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0.simple_rnn_cell_matmul_readvariableop_resource/simple_rnn_cell_biasadd_readvariableop_resource0simple_rnn_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ђ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_7846936*
condR
while_cond_7846935*9
output_shapes(
&: : : : :         ђ: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   ╠
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ђ*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  ђl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  ђ`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:         ђЦ
NoOpNoOp'^simple_rnn_cell/BiasAdd/ReadVariableOp&^simple_rnn_cell/MatMul/ReadVariableOp(^simple_rnn_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::                  :         ђ: : : 2P
&simple_rnn_cell/BiasAdd/ReadVariableOp&simple_rnn_cell/BiasAdd/ReadVariableOp2N
%simple_rnn_cell/MatMul/ReadVariableOp%simple_rnn_cell/MatMul/ReadVariableOp2R
'simple_rnn_cell/MatMul_1/ReadVariableOp'simple_rnn_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:YU
(
_output_shapes
:         ђ
)
_user_specified_nameinitial_state_0:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Я
Ј
E__inference_model_65_layer_call_and_return_conditional_losses_7846437
input_38
input_53(
simple_rnn_21_7846422:	ђ$
simple_rnn_21_7846424:	ђ)
simple_rnn_21_7846426:
ђђ#
dense_14_7846430:	ђ
dense_14_7846432:
identity

identity_1ѕб dense_14/StatefulPartitionedCallб%simple_rnn_21/StatefulPartitionedCallл
%simple_rnn_21/StatefulPartitionedCallStatefulPartitionedCallinput_38input_53simple_rnn_21_7846422simple_rnn_21_7846424simple_rnn_21_7846426*
Tin	
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:                  ђ:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7846421е
 dense_14/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_21/StatefulPartitionedCall:output:0dense_14_7846430dense_14_7846432*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_7846309Ё
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  ђ

Identity_1Identity.simple_rnn_21/StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ђm
NoOpNoOp!^dense_14/StatefulPartitionedCall&^simple_rnn_21/StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:                  :         ђ: : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2N
%simple_rnn_21/StatefulPartitionedCall%simple_rnn_21/StatefulPartitionedCall:'#
!
_user_specified_name	7846432:'#
!
_user_specified_name	7846430:'#
!
_user_specified_name	7846426:'#
!
_user_specified_name	7846424:'#
!
_user_specified_name	7846422:RN
(
_output_shapes
:         ђ
"
_user_specified_name
input_53:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
input_38
Ъ?
К
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7846801
inputs_0A
.simple_rnn_cell_matmul_readvariableop_resource:	ђ>
/simple_rnn_cell_biasadd_readvariableop_resource:	ђD
0simple_rnn_cell_matmul_1_readvariableop_resource:
ђђ
identity

identity_1ѕб&simple_rnn_cell/BiasAdd/ReadVariableOpб%simple_rnn_cell/MatMul/ReadVariableOpб'simple_rnn_cell/MatMul_1/ReadVariableOpбwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ђc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЋ
%simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp.simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0ю
simple_rnn_cell/MatMulMatMulstrided_slice_2:output:0-simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЊ
&simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp/simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Д
simple_rnn_cell/BiasAddBiasAdd simple_rnn_cell/MatMul:product:0.simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђџ
'simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp0simple_rnn_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0ќ
simple_rnn_cell/MatMul_1MatMulzeros:output:0/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
simple_rnn_cell/addAddV2 simple_rnn_cell/BiasAdd:output:0"simple_rnn_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђh
simple_rnn_cell/TanhTanhsimple_rnn_cell/add:z:0*
T0*(
_output_shapes
:         ђn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ╣
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0.simple_rnn_cell_matmul_readvariableop_resource/simple_rnn_cell_biasadd_readvariableop_resource0simple_rnn_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ђ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_7846734*
condR
while_cond_7846733*9
output_shapes(
&: : : : :         ђ: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   ╠
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ђ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  ђl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  ђ`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:         ђЦ
NoOpNoOp'^simple_rnn_cell/BiasAdd/ReadVariableOp&^simple_rnn_cell/MatMul/ReadVariableOp(^simple_rnn_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2P
&simple_rnn_cell/BiasAdd/ReadVariableOp&simple_rnn_cell/BiasAdd/ReadVariableOp2N
%simple_rnn_cell/MatMul/ReadVariableOp%simple_rnn_cell/MatMul/ReadVariableOp2R
'simple_rnn_cell/MatMul_1/ReadVariableOp'simple_rnn_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
Є.
╗
while_body_7846354
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0I
6while_simple_rnn_cell_matmul_readvariableop_resource_0:	ђF
7while_simple_rnn_cell_biasadd_readvariableop_resource_0:	ђL
8while_simple_rnn_cell_matmul_1_readvariableop_resource_0:
ђђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorG
4while_simple_rnn_cell_matmul_readvariableop_resource:	ђD
5while_simple_rnn_cell_biasadd_readvariableop_resource:	ђJ
6while_simple_rnn_cell_matmul_1_readvariableop_resource:
ђђѕб,while/simple_rnn_cell/BiasAdd/ReadVariableOpб+while/simple_rnn_cell/MatMul/ReadVariableOpб-while/simple_rnn_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Б
+while/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp6while_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0└
while/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:03while/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђА
,while/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp7while_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0╣
while/simple_rnn_cell/BiasAddBiasAdd&while/simple_rnn_cell/MatMul:product:04while/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђе
-while/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp8while_simple_rnn_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0Д
while/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_25while/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђД
while/simple_rnn_cell/addAddV2&while/simple_rnn_cell/BiasAdd:output:0(while/simple_rnn_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђt
while/simple_rnn_cell/TanhTanhwhile/simple_rnn_cell/add:z:0*
T0*(
_output_shapes
:         ђК
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/simple_rnn_cell/Tanh:y:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: |
while/Identity_4Identitywhile/simple_rnn_cell/Tanh:y:0^while/NoOp*
T0*(
_output_shapes
:         ђх

while/NoOpNoOp-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"p
5while_simple_rnn_cell_biasadd_readvariableop_resource7while_simple_rnn_cell_biasadd_readvariableop_resource_0"r
6while_simple_rnn_cell_matmul_1_readvariableop_resource8while_simple_rnn_cell_matmul_1_readvariableop_resource_0"n
4while_simple_rnn_cell_matmul_readvariableop_resource6while_simple_rnn_cell_matmul_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ђ: : : : : 2\
,while/simple_rnn_cell/BiasAdd/ReadVariableOp,while/simple_rnn_cell/BiasAdd/ReadVariableOp2Z
+while/simple_rnn_cell/MatMul/ReadVariableOp+while/simple_rnn_cell/MatMul/ReadVariableOp2^
-while/simple_rnn_cell/MatMul_1/ReadVariableOp-while/simple_rnn_cell/MatMul_1/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ь
»
while_cond_7846624
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_7846624___redundant_placeholder05
1while_while_cond_7846624___redundant_placeholder15
1while_while_cond_7846624___redundant_placeholder25
1while_while_cond_7846624___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ђ: :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ш5
г
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7846112

inputs*
simple_rnn_cell_7846036:	ђ&
simple_rnn_cell_7846038:	ђ+
simple_rnn_cell_7846040:
ђђ
identity

identity_1ѕб'simple_rnn_cell/StatefulPartitionedCallбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ђc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskс
'simple_rnn_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_7846036simple_rnn_cell_7846038simple_rnn_cell_7846040*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         ђ:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_7846035n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ы
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_7846036simple_rnn_cell_7846038simple_rnn_cell_7846040*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ђ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_7846048*
condR
while_cond_7846047*9
output_shapes(
&: : : : :         ђ: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   ╠
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ђ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  ђl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  ђ`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:         ђT
NoOpNoOp(^simple_rnn_cell/StatefulPartitionedCall^while*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2R
'simple_rnn_cell/StatefulPartitionedCall'simple_rnn_cell/StatefulPartitionedCall2
whilewhile:'#
!
_user_specified_name	7846040:'#
!
_user_specified_name	7846038:'#
!
_user_specified_name	7846036:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
ў
П
1__inference_simple_rnn_cell_layer_call_fn_7847057

inputs
states_0
unknown:	ђ
	unknown_0:	ђ
	unknown_1:
ђђ
identity

identity_1ѕбStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         ђ:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_7845915p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ђ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:         :         ђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	7847051:'#
!
_user_specified_name	7847049:'#
!
_user_specified_name	7847047:RN
(
_output_shapes
:         ђ
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Т
в
L__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_7846035

inputs

states1
matmul_readvariableop_resource:	ђ.
biasadd_readvariableop_resource:	ђ4
 matmul_1_readvariableop_resource:
ђђ
identity

identity_1ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђH
TanhTanhadd:z:0*
T0*(
_output_shapes
:         ђX
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:         ђZ

Identity_1IdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:         ђm
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:         :         ђ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:PL
(
_output_shapes
:         ђ
 
_user_specified_namestates:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
П
Р
/__inference_simple_rnn_21_layer_call_fn_7846569

inputs
initial_state_0
unknown:	ђ
	unknown_0:	ђ
	unknown_1:
ђђ
identity

identity_1ѕбStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsinitial_state_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:                  ђ:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7846270}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  ђr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ђ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::                  :         ђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	7846563:'#
!
_user_specified_name	7846561:'#
!
_user_specified_name	7846559:YU
(
_output_shapes
:         ђ
)
_user_specified_nameinitial_state_0:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Ї#
м
while_body_7846048
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_simple_rnn_cell_7846070_0:	ђ.
while_simple_rnn_cell_7846072_0:	ђ3
while_simple_rnn_cell_7846074_0:
ђђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_7846070:	ђ,
while_simple_rnn_cell_7846072:	ђ1
while_simple_rnn_cell_7846074:
ђђѕб-while/simple_rnn_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ъ
-while/simple_rnn_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_7846070_0while_simple_rnn_cell_7846072_0while_simple_rnn_cell_7846074_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         ђ:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_7846035▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder6while/simple_rnn_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ћ
while/Identity_4Identity6while/simple_rnn_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         ђX

while/NoOpNoOp.^while/simple_rnn_cell/StatefulPartitionedCall*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"@
while_simple_rnn_cell_7846070while_simple_rnn_cell_7846070_0"@
while_simple_rnn_cell_7846072while_simple_rnn_cell_7846072_0"@
while_simple_rnn_cell_7846074while_simple_rnn_cell_7846074_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ђ: : : : : 2^
-while/simple_rnn_cell/StatefulPartitionedCall-while/simple_rnn_cell/StatefulPartitionedCall:'	#
!
_user_specified_name	7846074:'#
!
_user_specified_name	7846072:'#
!
_user_specified_name	7846070:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ш5
г
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7845992

inputs*
simple_rnn_cell_7845916:	ђ&
simple_rnn_cell_7845918:	ђ+
simple_rnn_cell_7845920:
ђђ
identity

identity_1ѕб'simple_rnn_cell/StatefulPartitionedCallбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ђc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskс
'simple_rnn_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_7845916simple_rnn_cell_7845918simple_rnn_cell_7845920*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         ђ:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_7845915n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ы
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_7845916simple_rnn_cell_7845918simple_rnn_cell_7845920*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ђ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_7845928*
condR
while_cond_7845927*9
output_shapes(
&: : : : :         ђ: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   ╠
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ђ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  ђl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  ђ`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:         ђT
NoOpNoOp(^simple_rnn_cell/StatefulPartitionedCall^while*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2R
'simple_rnn_cell/StatefulPartitionedCall'simple_rnn_cell/StatefulPartitionedCall2
whilewhile:'#
!
_user_specified_name	7845920:'#
!
_user_specified_name	7845918:'#
!
_user_specified_name	7845916:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
═

¤
/__inference_simple_rnn_21_layer_call_fn_7846555
inputs_0
unknown:	ђ
	unknown_0:	ђ
	unknown_1:
ђђ
identity

identity_1ѕбStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:                  ђ:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7846112}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  ђr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ђ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	7846549:'#
!
_user_specified_name	7846547:'#
!
_user_specified_name	7846545:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
Я
Ј
E__inference_model_65_layer_call_and_return_conditional_losses_7846317
input_38
input_53(
simple_rnn_21_7846271:	ђ$
simple_rnn_21_7846273:	ђ)
simple_rnn_21_7846275:
ђђ#
dense_14_7846310:	ђ
dense_14_7846312:
identity

identity_1ѕб dense_14/StatefulPartitionedCallб%simple_rnn_21/StatefulPartitionedCallл
%simple_rnn_21/StatefulPartitionedCallStatefulPartitionedCallinput_38input_53simple_rnn_21_7846271simple_rnn_21_7846273simple_rnn_21_7846275*
Tin	
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:                  ђ:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7846270е
 dense_14/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_21/StatefulPartitionedCall:output:0dense_14_7846310dense_14_7846312*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_7846309Ё
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  ђ

Identity_1Identity.simple_rnn_21/StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ђm
NoOpNoOp!^dense_14/StatefulPartitionedCall&^simple_rnn_21/StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:                  :         ђ: : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2N
%simple_rnn_21/StatefulPartitionedCall%simple_rnn_21/StatefulPartitionedCall:'#
!
_user_specified_name	7846312:'#
!
_user_specified_name	7846310:'#
!
_user_specified_name	7846275:'#
!
_user_specified_name	7846273:'#
!
_user_specified_name	7846271:RN
(
_output_shapes
:         ђ
"
_user_specified_name
input_53:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
input_38
═

¤
/__inference_simple_rnn_21_layer_call_fn_7846542
inputs_0
unknown:	ђ
	unknown_0:	ђ
	unknown_1:
ђђ
identity

identity_1ѕбStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:                  ђ:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7845992}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  ђr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ђ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	7846536:'#
!
_user_specified_name	7846534:'#
!
_user_specified_name	7846532:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
Џ
Р
)model_65_simple_rnn_21_while_cond_7845777J
Fmodel_65_simple_rnn_21_while_model_65_simple_rnn_21_while_loop_counterP
Lmodel_65_simple_rnn_21_while_model_65_simple_rnn_21_while_maximum_iterations,
(model_65_simple_rnn_21_while_placeholder.
*model_65_simple_rnn_21_while_placeholder_1.
*model_65_simple_rnn_21_while_placeholder_2J
Fmodel_65_simple_rnn_21_while_less_model_65_simple_rnn_21_strided_slicec
_model_65_simple_rnn_21_while_model_65_simple_rnn_21_while_cond_7845777___redundant_placeholder0c
_model_65_simple_rnn_21_while_model_65_simple_rnn_21_while_cond_7845777___redundant_placeholder1c
_model_65_simple_rnn_21_while_model_65_simple_rnn_21_while_cond_7845777___redundant_placeholder2c
_model_65_simple_rnn_21_while_model_65_simple_rnn_21_while_cond_7845777___redundant_placeholder3)
%model_65_simple_rnn_21_while_identity
╝
!model_65/simple_rnn_21/while/LessLess(model_65_simple_rnn_21_while_placeholderFmodel_65_simple_rnn_21_while_less_model_65_simple_rnn_21_strided_slice*
T0*
_output_shapes
: y
%model_65/simple_rnn_21/while/IdentityIdentity%model_65/simple_rnn_21/while/Less:z:0*
T0
*
_output_shapes
: "W
%model_65_simple_rnn_21_while_identity.model_65/simple_rnn_21/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ђ: :::::

_output_shapes
::\X

_output_shapes
: 
>
_user_specified_name&$model_65/simple_rnn_21/strided_slice:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :gc

_output_shapes
: 
I
_user_specified_name1/model_65/simple_rnn_21/while/maximum_iterations:a ]

_output_shapes
: 
C
_user_specified_name+)model_65/simple_rnn_21/while/loop_counter
ж
Ј
*__inference_model_65_layer_call_fn_7846455
input_38
input_53
unknown:	ђ
	unknown_0:	ђ
	unknown_1:
ђђ
	unknown_2:	ђ
	unknown_3:
identity

identity_1ѕбStatefulPartitionedCall░
StatefulPartitionedCallStatefulPartitionedCallinput_38input_53unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2*
Tout
2*
_collective_manager_ids
 *H
_output_shapes6
4:                  :         ђ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_model_65_layer_call_and_return_conditional_losses_7846317|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ђ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:                  :         ђ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	7846449:'#
!
_user_specified_name	7846447:'#
!
_user_specified_name	7846445:'#
!
_user_specified_name	7846443:'#
!
_user_specified_name	7846441:RN
(
_output_shapes
:         ђ
"
_user_specified_name
input_53:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
input_38
┴
і
%__inference_signature_wrapper_7846529
input_38
input_53
unknown:	ђ
	unknown_0:	ђ
	unknown_1:
ђђ
	unknown_2:	ђ
	unknown_3:
identity

identity_1ѕбStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinput_38input_53unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2*
Tout
2*
_collective_manager_ids
 *H
_output_shapes6
4:                  :         ђ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *+
f&R$
"__inference__wrapped_model_7845872|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ђ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:                  :         ђ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	7846523:'#
!
_user_specified_name	7846521:'#
!
_user_specified_name	7846519:'#
!
_user_specified_name	7846517:'#
!
_user_specified_name	7846515:RN
(
_output_shapes
:         ђ
"
_user_specified_name
input_53:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
input_38
ж
Ј
*__inference_model_65_layer_call_fn_7846473
input_38
input_53
unknown:	ђ
	unknown_0:	ђ
	unknown_1:
ђђ
	unknown_2:	ђ
	unknown_3:
identity

identity_1ѕбStatefulPartitionedCall░
StatefulPartitionedCallStatefulPartitionedCallinput_38input_53unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2*
Tout
2*
_collective_manager_ids
 *H
_output_shapes6
4:                  :         ђ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_model_65_layer_call_and_return_conditional_losses_7846437|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ђ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:                  :         ђ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	7846467:'#
!
_user_specified_name	7846465:'#
!
_user_specified_name	7846463:'#
!
_user_specified_name	7846461:'#
!
_user_specified_name	7846459:RN
(
_output_shapes
:         ђ
"
_user_specified_name
input_53:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
input_38
Є.
╗
while_body_7846936
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0I
6while_simple_rnn_cell_matmul_readvariableop_resource_0:	ђF
7while_simple_rnn_cell_biasadd_readvariableop_resource_0:	ђL
8while_simple_rnn_cell_matmul_1_readvariableop_resource_0:
ђђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorG
4while_simple_rnn_cell_matmul_readvariableop_resource:	ђD
5while_simple_rnn_cell_biasadd_readvariableop_resource:	ђJ
6while_simple_rnn_cell_matmul_1_readvariableop_resource:
ђђѕб,while/simple_rnn_cell/BiasAdd/ReadVariableOpб+while/simple_rnn_cell/MatMul/ReadVariableOpб-while/simple_rnn_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Б
+while/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp6while_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0└
while/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:03while/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђА
,while/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp7while_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0╣
while/simple_rnn_cell/BiasAddBiasAdd&while/simple_rnn_cell/MatMul:product:04while/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђе
-while/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp8while_simple_rnn_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0Д
while/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_25while/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђД
while/simple_rnn_cell/addAddV2&while/simple_rnn_cell/BiasAdd:output:0(while/simple_rnn_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђt
while/simple_rnn_cell/TanhTanhwhile/simple_rnn_cell/add:z:0*
T0*(
_output_shapes
:         ђК
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/simple_rnn_cell/Tanh:y:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: |
while/Identity_4Identitywhile/simple_rnn_cell/Tanh:y:0^while/NoOp*
T0*(
_output_shapes
:         ђх

while/NoOpNoOp-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"p
5while_simple_rnn_cell_biasadd_readvariableop_resource7while_simple_rnn_cell_biasadd_readvariableop_resource_0"r
6while_simple_rnn_cell_matmul_1_readvariableop_resource8while_simple_rnn_cell_matmul_1_readvariableop_resource_0"n
4while_simple_rnn_cell_matmul_readvariableop_resource6while_simple_rnn_cell_matmul_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ђ: : : : : 2\
,while/simple_rnn_cell/BiasAdd/ReadVariableOp,while/simple_rnn_cell/BiasAdd/ReadVariableOp2Z
+while/simple_rnn_cell/MatMul/ReadVariableOp+while/simple_rnn_cell/MatMul/ReadVariableOp2^
-while/simple_rnn_cell/MatMul_1/ReadVariableOp-while/simple_rnn_cell/MatMul_1/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
П
Р
/__inference_simple_rnn_21_layer_call_fn_7846583

inputs
initial_state_0
unknown:	ђ
	unknown_0:	ђ
	unknown_1:
ђђ
identity

identity_1ѕбStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsinitial_state_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:                  ђ:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7846421}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  ђr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ђ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::                  :         ђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	7846577:'#
!
_user_specified_name	7846575:'#
!
_user_specified_name	7846573:YU
(
_output_shapes
:         ђ
)
_user_specified_nameinitial_state_0:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
У
Г
while_cond_7846353
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_7846353___redundant_placeholder05
1while_while_cond_7846353___redundant_placeholder15
1while_while_cond_7846353___redundant_placeholder25
1while_while_cond_7846353___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ђ: :::::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
В
ь
L__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_7847088

inputs
states_01
matmul_readvariableop_resource:	ђ.
biasadd_readvariableop_resource:	ђ4
 matmul_1_readvariableop_resource:
ђђ
identity

identity_1ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђH
TanhTanhadd:z:0*
T0*(
_output_shapes
:         ђX
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:         ђZ

Identity_1IdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:         ђm
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:         :         ђ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
(
_output_shapes
:         ђ
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ь
»
while_cond_7846047
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_7846047___redundant_placeholder05
1while_while_cond_7846047___redundant_placeholder15
1while_while_cond_7846047___redundant_placeholder25
1while_while_cond_7846047___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ђ: :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Е
ў
*__inference_dense_14_layer_call_fn_7847012

inputs
unknown:	ђ
	unknown_0:
identityѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_7846309|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	7847008:'#
!
_user_specified_name	7847006:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs
╝8
┌
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7846902

inputs
initial_state_0A
.simple_rnn_cell_matmul_readvariableop_resource:	ђ>
/simple_rnn_cell_biasadd_readvariableop_resource:	ђD
0simple_rnn_cell_matmul_1_readvariableop_resource:
ђђ
identity

identity_1ѕб&simple_rnn_cell/BiasAdd/ReadVariableOpб%simple_rnn_cell/MatMul/ReadVariableOpб'simple_rnn_cell/MatMul_1/ReadVariableOpбwhilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  P
ShapeShapetranspose:y:0*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЋ
%simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp.simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0ю
simple_rnn_cell/MatMulMatMulstrided_slice_1:output:0-simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЊ
&simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp/simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Д
simple_rnn_cell/BiasAddBiasAdd simple_rnn_cell/MatMul:product:0.simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђџ
'simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp0simple_rnn_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0Ќ
simple_rnn_cell/MatMul_1MatMulinitial_state_0/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
simple_rnn_cell/addAddV2 simple_rnn_cell/BiasAdd:output:0"simple_rnn_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђh
simple_rnn_cell/TanhTanhsimple_rnn_cell/add:z:0*
T0*(
_output_shapes
:         ђn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : И
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0initial_state_0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0.simple_rnn_cell_matmul_readvariableop_resource/simple_rnn_cell_biasadd_readvariableop_resource0simple_rnn_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ђ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_7846835*
condR
while_cond_7846834*9
output_shapes(
&: : : : :         ђ: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   ╠
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ђ*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  ђl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  ђ`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:         ђЦ
NoOpNoOp'^simple_rnn_cell/BiasAdd/ReadVariableOp&^simple_rnn_cell/MatMul/ReadVariableOp(^simple_rnn_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::                  :         ђ: : : 2P
&simple_rnn_cell/BiasAdd/ReadVariableOp&simple_rnn_cell/BiasAdd/ReadVariableOp2N
%simple_rnn_cell/MatMul/ReadVariableOp%simple_rnn_cell/MatMul/ReadVariableOp2R
'simple_rnn_cell/MatMul_1/ReadVariableOp'simple_rnn_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:YU
(
_output_shapes
:         ђ
)
_user_specified_nameinitial_state_0:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
У
Г
while_cond_7846935
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_7846935___redundant_placeholder05
1while_while_cond_7846935___redundant_placeholder15
1while_while_cond_7846935___redundant_placeholder25
1while_while_cond_7846935___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ђ: :::::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
ў
П
1__inference_simple_rnn_cell_layer_call_fn_7847071

inputs
states_0
unknown:	ђ
	unknown_0:	ђ
	unknown_1:
ђђ
identity

identity_1ѕбStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         ђ:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_7846035p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ђ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:         :         ђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	7847065:'#
!
_user_specified_name	7847063:'#
!
_user_specified_name	7847061:RN
(
_output_shapes
:         ђ
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
В
ь
L__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_7847105

inputs
states_01
matmul_readvariableop_resource:	ђ.
biasadd_readvariableop_resource:	ђ4
 matmul_1_readvariableop_resource:
ђђ
identity

identity_1ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђH
TanhTanhadd:z:0*
T0*(
_output_shapes
:         ђX
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:         ђZ

Identity_1IdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:         ђm
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:         :         ђ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
(
_output_shapes
:         ђ
"
_user_specified_name
states_0:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ъ?
К
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7846692
inputs_0A
.simple_rnn_cell_matmul_readvariableop_resource:	ђ>
/simple_rnn_cell_biasadd_readvariableop_resource:	ђD
0simple_rnn_cell_matmul_1_readvariableop_resource:
ђђ
identity

identity_1ѕб&simple_rnn_cell/BiasAdd/ReadVariableOpб%simple_rnn_cell/MatMul/ReadVariableOpб'simple_rnn_cell/MatMul_1/ReadVariableOpбwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ђs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ђc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЋ
%simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp.simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0ю
simple_rnn_cell/MatMulMatMulstrided_slice_2:output:0-simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЊ
&simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp/simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Д
simple_rnn_cell/BiasAddBiasAdd simple_rnn_cell/MatMul:product:0.simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђџ
'simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp0simple_rnn_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0ќ
simple_rnn_cell/MatMul_1MatMulzeros:output:0/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
simple_rnn_cell/addAddV2 simple_rnn_cell/BiasAdd:output:0"simple_rnn_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђh
simple_rnn_cell/TanhTanhsimple_rnn_cell/add:z:0*
T0*(
_output_shapes
:         ђn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ╣
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0.simple_rnn_cell_matmul_readvariableop_resource/simple_rnn_cell_biasadd_readvariableop_resource0simple_rnn_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ђ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_7846625*
condR
while_cond_7846624*9
output_shapes(
&: : : : :         ђ: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   ╠
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ђ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  ђl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  ђ`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:         ђЦ
NoOpNoOp'^simple_rnn_cell/BiasAdd/ReadVariableOp&^simple_rnn_cell/MatMul/ReadVariableOp(^simple_rnn_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2P
&simple_rnn_cell/BiasAdd/ReadVariableOp&simple_rnn_cell/BiasAdd/ReadVariableOp2N
%simple_rnn_cell/MatMul/ReadVariableOp%simple_rnn_cell/MatMul/ReadVariableOp2R
'simple_rnn_cell/MatMul_1/ReadVariableOp'simple_rnn_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
Ь
»
while_cond_7845927
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_7845927___redundant_placeholder05
1while_while_cond_7845927___redundant_placeholder15
1while_while_cond_7845927___redundant_placeholder25
1while_while_cond_7845927___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ђ: :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Т
в
L__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_7845915

inputs

states1
matmul_readvariableop_resource:	ђ.
biasadd_readvariableop_resource:	ђ4
 matmul_1_readvariableop_resource:
ђђ
identity

identity_1ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђH
TanhTanhadd:z:0*
T0*(
_output_shapes
:         ђX
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:         ђZ

Identity_1IdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:         ђm
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:         :         ђ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:PL
(
_output_shapes
:         ђ
 
_user_specified_namestates:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┤8
п
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7846421

inputs
initial_stateA
.simple_rnn_cell_matmul_readvariableop_resource:	ђ>
/simple_rnn_cell_biasadd_readvariableop_resource:	ђD
0simple_rnn_cell_matmul_1_readvariableop_resource:
ђђ
identity

identity_1ѕб&simple_rnn_cell/BiasAdd/ReadVariableOpб%simple_rnn_cell/MatMul/ReadVariableOpб'simple_rnn_cell/MatMul_1/ReadVariableOpбwhilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  P
ShapeShapetranspose:y:0*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЋ
%simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp.simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0ю
simple_rnn_cell/MatMulMatMulstrided_slice_1:output:0-simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЊ
&simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp/simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Д
simple_rnn_cell/BiasAddBiasAdd simple_rnn_cell/MatMul:product:0.simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђџ
'simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp0simple_rnn_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0Ћ
simple_rnn_cell/MatMul_1MatMulinitial_state/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
simple_rnn_cell/addAddV2 simple_rnn_cell/BiasAdd:output:0"simple_rnn_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђh
simple_rnn_cell/TanhTanhsimple_rnn_cell/add:z:0*
T0*(
_output_shapes
:         ђn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Х
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0initial_statestrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0.simple_rnn_cell_matmul_readvariableop_resource/simple_rnn_cell_biasadd_readvariableop_resource0simple_rnn_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ђ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_7846354*
condR
while_cond_7846353*9
output_shapes(
&: : : : :         ђ: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   ╠
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ђ*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  ђl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  ђ`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:         ђЦ
NoOpNoOp'^simple_rnn_cell/BiasAdd/ReadVariableOp&^simple_rnn_cell/MatMul/ReadVariableOp(^simple_rnn_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::                  :         ђ: : : 2P
&simple_rnn_cell/BiasAdd/ReadVariableOp&simple_rnn_cell/BiasAdd/ReadVariableOp2N
%simple_rnn_cell/MatMul/ReadVariableOp%simple_rnn_cell/MatMul/ReadVariableOp2R
'simple_rnn_cell/MatMul_1/ReadVariableOp'simple_rnn_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:WS
(
_output_shapes
:         ђ
'
_user_specified_nameinitial_state:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
ДD
­
)model_65_simple_rnn_21_while_body_7845778J
Fmodel_65_simple_rnn_21_while_model_65_simple_rnn_21_while_loop_counterP
Lmodel_65_simple_rnn_21_while_model_65_simple_rnn_21_while_maximum_iterations,
(model_65_simple_rnn_21_while_placeholder.
*model_65_simple_rnn_21_while_placeholder_1.
*model_65_simple_rnn_21_while_placeholder_2G
Cmodel_65_simple_rnn_21_while_model_65_simple_rnn_21_strided_slice_0є
Ђmodel_65_simple_rnn_21_while_tensorarrayv2read_tensorlistgetitem_model_65_simple_rnn_21_tensorarrayunstack_tensorlistfromtensor_0`
Mmodel_65_simple_rnn_21_while_simple_rnn_cell_matmul_readvariableop_resource_0:	ђ]
Nmodel_65_simple_rnn_21_while_simple_rnn_cell_biasadd_readvariableop_resource_0:	ђc
Omodel_65_simple_rnn_21_while_simple_rnn_cell_matmul_1_readvariableop_resource_0:
ђђ)
%model_65_simple_rnn_21_while_identity+
'model_65_simple_rnn_21_while_identity_1+
'model_65_simple_rnn_21_while_identity_2+
'model_65_simple_rnn_21_while_identity_3+
'model_65_simple_rnn_21_while_identity_4E
Amodel_65_simple_rnn_21_while_model_65_simple_rnn_21_strided_sliceЃ
model_65_simple_rnn_21_while_tensorarrayv2read_tensorlistgetitem_model_65_simple_rnn_21_tensorarrayunstack_tensorlistfromtensor^
Kmodel_65_simple_rnn_21_while_simple_rnn_cell_matmul_readvariableop_resource:	ђ[
Lmodel_65_simple_rnn_21_while_simple_rnn_cell_biasadd_readvariableop_resource:	ђa
Mmodel_65_simple_rnn_21_while_simple_rnn_cell_matmul_1_readvariableop_resource:
ђђѕбCmodel_65/simple_rnn_21/while/simple_rnn_cell/BiasAdd/ReadVariableOpбBmodel_65/simple_rnn_21/while/simple_rnn_cell/MatMul/ReadVariableOpбDmodel_65/simple_rnn_21/while/simple_rnn_cell/MatMul_1/ReadVariableOpЪ
Nmodel_65/simple_rnn_21/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       џ
@model_65/simple_rnn_21/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЂmodel_65_simple_rnn_21_while_tensorarrayv2read_tensorlistgetitem_model_65_simple_rnn_21_tensorarrayunstack_tensorlistfromtensor_0(model_65_simple_rnn_21_while_placeholderWmodel_65/simple_rnn_21/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Л
Bmodel_65/simple_rnn_21/while/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOpMmodel_65_simple_rnn_21_while_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0Ё
3model_65/simple_rnn_21/while/simple_rnn_cell/MatMulMatMulGmodel_65/simple_rnn_21/while/TensorArrayV2Read/TensorListGetItem:item:0Jmodel_65/simple_rnn_21/while/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ¤
Cmodel_65/simple_rnn_21/while/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpNmodel_65_simple_rnn_21_while_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0■
4model_65/simple_rnn_21/while/simple_rnn_cell/BiasAddBiasAdd=model_65/simple_rnn_21/while/simple_rnn_cell/MatMul:product:0Kmodel_65/simple_rnn_21/while/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђо
Dmodel_65/simple_rnn_21/while/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpOmodel_65_simple_rnn_21_while_simple_rnn_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0В
5model_65/simple_rnn_21/while/simple_rnn_cell/MatMul_1MatMul*model_65_simple_rnn_21_while_placeholder_2Lmodel_65/simple_rnn_21/while/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђВ
0model_65/simple_rnn_21/while/simple_rnn_cell/addAddV2=model_65/simple_rnn_21/while/simple_rnn_cell/BiasAdd:output:0?model_65/simple_rnn_21/while/simple_rnn_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђб
1model_65/simple_rnn_21/while/simple_rnn_cell/TanhTanh4model_65/simple_rnn_21/while/simple_rnn_cell/add:z:0*
T0*(
_output_shapes
:         ђБ
Amodel_65/simple_rnn_21/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*model_65_simple_rnn_21_while_placeholder_1(model_65_simple_rnn_21_while_placeholder5model_65/simple_rnn_21/while/simple_rnn_cell/Tanh:y:0*
_output_shapes
: *
element_dtype0:жУмd
"model_65/simple_rnn_21/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :А
 model_65/simple_rnn_21/while/addAddV2(model_65_simple_rnn_21_while_placeholder+model_65/simple_rnn_21/while/add/y:output:0*
T0*
_output_shapes
: f
$model_65/simple_rnn_21/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :├
"model_65/simple_rnn_21/while/add_1AddV2Fmodel_65_simple_rnn_21_while_model_65_simple_rnn_21_while_loop_counter-model_65/simple_rnn_21/while/add_1/y:output:0*
T0*
_output_shapes
: ъ
%model_65/simple_rnn_21/while/IdentityIdentity&model_65/simple_rnn_21/while/add_1:z:0"^model_65/simple_rnn_21/while/NoOp*
T0*
_output_shapes
: к
'model_65/simple_rnn_21/while/Identity_1IdentityLmodel_65_simple_rnn_21_while_model_65_simple_rnn_21_while_maximum_iterations"^model_65/simple_rnn_21/while/NoOp*
T0*
_output_shapes
: ъ
'model_65/simple_rnn_21/while/Identity_2Identity$model_65/simple_rnn_21/while/add:z:0"^model_65/simple_rnn_21/while/NoOp*
T0*
_output_shapes
: ╦
'model_65/simple_rnn_21/while/Identity_3IdentityQmodel_65/simple_rnn_21/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^model_65/simple_rnn_21/while/NoOp*
T0*
_output_shapes
: ┴
'model_65/simple_rnn_21/while/Identity_4Identity5model_65/simple_rnn_21/while/simple_rnn_cell/Tanh:y:0"^model_65/simple_rnn_21/while/NoOp*
T0*(
_output_shapes
:         ђЉ
!model_65/simple_rnn_21/while/NoOpNoOpD^model_65/simple_rnn_21/while/simple_rnn_cell/BiasAdd/ReadVariableOpC^model_65/simple_rnn_21/while/simple_rnn_cell/MatMul/ReadVariableOpE^model_65/simple_rnn_21/while/simple_rnn_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "[
'model_65_simple_rnn_21_while_identity_10model_65/simple_rnn_21/while/Identity_1:output:0"[
'model_65_simple_rnn_21_while_identity_20model_65/simple_rnn_21/while/Identity_2:output:0"[
'model_65_simple_rnn_21_while_identity_30model_65/simple_rnn_21/while/Identity_3:output:0"[
'model_65_simple_rnn_21_while_identity_40model_65/simple_rnn_21/while/Identity_4:output:0"W
%model_65_simple_rnn_21_while_identity.model_65/simple_rnn_21/while/Identity:output:0"ѕ
Amodel_65_simple_rnn_21_while_model_65_simple_rnn_21_strided_sliceCmodel_65_simple_rnn_21_while_model_65_simple_rnn_21_strided_slice_0"ъ
Lmodel_65_simple_rnn_21_while_simple_rnn_cell_biasadd_readvariableop_resourceNmodel_65_simple_rnn_21_while_simple_rnn_cell_biasadd_readvariableop_resource_0"а
Mmodel_65_simple_rnn_21_while_simple_rnn_cell_matmul_1_readvariableop_resourceOmodel_65_simple_rnn_21_while_simple_rnn_cell_matmul_1_readvariableop_resource_0"ю
Kmodel_65_simple_rnn_21_while_simple_rnn_cell_matmul_readvariableop_resourceMmodel_65_simple_rnn_21_while_simple_rnn_cell_matmul_readvariableop_resource_0"Ё
model_65_simple_rnn_21_while_tensorarrayv2read_tensorlistgetitem_model_65_simple_rnn_21_tensorarrayunstack_tensorlistfromtensorЂmodel_65_simple_rnn_21_while_tensorarrayv2read_tensorlistgetitem_model_65_simple_rnn_21_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ђ: : : : : 2і
Cmodel_65/simple_rnn_21/while/simple_rnn_cell/BiasAdd/ReadVariableOpCmodel_65/simple_rnn_21/while/simple_rnn_cell/BiasAdd/ReadVariableOp2ѕ
Bmodel_65/simple_rnn_21/while/simple_rnn_cell/MatMul/ReadVariableOpBmodel_65/simple_rnn_21/while/simple_rnn_cell/MatMul/ReadVariableOp2ї
Dmodel_65/simple_rnn_21/while/simple_rnn_cell/MatMul_1/ReadVariableOpDmodel_65/simple_rnn_21/while/simple_rnn_cell/MatMul_1/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:vr

_output_shapes
: 
X
_user_specified_name@>model_65/simple_rnn_21/TensorArrayUnstack/TensorListFromTensor:\X

_output_shapes
: 
>
_user_specified_name&$model_65/simple_rnn_21/strided_slice:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :gc

_output_shapes
: 
I
_user_specified_name1/model_65/simple_rnn_21/while/maximum_iterations:a ]

_output_shapes
: 
C
_user_specified_name+)model_65/simple_rnn_21/while/loop_counter
Є.
╗
while_body_7846203
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0I
6while_simple_rnn_cell_matmul_readvariableop_resource_0:	ђF
7while_simple_rnn_cell_biasadd_readvariableop_resource_0:	ђL
8while_simple_rnn_cell_matmul_1_readvariableop_resource_0:
ђђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorG
4while_simple_rnn_cell_matmul_readvariableop_resource:	ђD
5while_simple_rnn_cell_biasadd_readvariableop_resource:	ђJ
6while_simple_rnn_cell_matmul_1_readvariableop_resource:
ђђѕб,while/simple_rnn_cell/BiasAdd/ReadVariableOpб+while/simple_rnn_cell/MatMul/ReadVariableOpб-while/simple_rnn_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Б
+while/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp6while_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0└
while/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:03while/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђА
,while/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp7while_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0╣
while/simple_rnn_cell/BiasAddBiasAdd&while/simple_rnn_cell/MatMul:product:04while/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђе
-while/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp8while_simple_rnn_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0Д
while/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_25while/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђД
while/simple_rnn_cell/addAddV2&while/simple_rnn_cell/BiasAdd:output:0(while/simple_rnn_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђt
while/simple_rnn_cell/TanhTanhwhile/simple_rnn_cell/add:z:0*
T0*(
_output_shapes
:         ђК
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/simple_rnn_cell/Tanh:y:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: |
while/Identity_4Identitywhile/simple_rnn_cell/Tanh:y:0^while/NoOp*
T0*(
_output_shapes
:         ђх

while/NoOpNoOp-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"p
5while_simple_rnn_cell_biasadd_readvariableop_resource7while_simple_rnn_cell_biasadd_readvariableop_resource_0"r
6while_simple_rnn_cell_matmul_1_readvariableop_resource8while_simple_rnn_cell_matmul_1_readvariableop_resource_0"n
4while_simple_rnn_cell_matmul_readvariableop_resource6while_simple_rnn_cell_matmul_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ђ: : : : : 2\
,while/simple_rnn_cell/BiasAdd/ReadVariableOp,while/simple_rnn_cell/BiasAdd/ReadVariableOp2Z
+while/simple_rnn_cell/MatMul/ReadVariableOp+while/simple_rnn_cell/MatMul/ReadVariableOp2^
-while/simple_rnn_cell/MatMul_1/ReadVariableOp-while/simple_rnn_cell/MatMul_1/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
ёo
»
"__inference__wrapped_model_7845872
input_38
input_53X
Emodel_65_simple_rnn_21_simple_rnn_cell_matmul_readvariableop_resource:	ђU
Fmodel_65_simple_rnn_21_simple_rnn_cell_biasadd_readvariableop_resource:	ђ[
Gmodel_65_simple_rnn_21_simple_rnn_cell_matmul_1_readvariableop_resource:
ђђF
3model_65_dense_14_tensordot_readvariableop_resource:	ђ?
1model_65_dense_14_biasadd_readvariableop_resource:
identity

identity_1ѕб(model_65/dense_14/BiasAdd/ReadVariableOpб*model_65/dense_14/Tensordot/ReadVariableOpб=model_65/simple_rnn_21/simple_rnn_cell/BiasAdd/ReadVariableOpб<model_65/simple_rnn_21/simple_rnn_cell/MatMul/ReadVariableOpб>model_65/simple_rnn_21/simple_rnn_cell/MatMul_1/ReadVariableOpбmodel_65/simple_rnn_21/whilez
%model_65/simple_rnn_21/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          д
 model_65/simple_rnn_21/transpose	Transposeinput_38.model_65/simple_rnn_21/transpose/perm:output:0*
T0*4
_output_shapes"
 :                  ~
model_65/simple_rnn_21/ShapeShape$model_65/simple_rnn_21/transpose:y:0*
T0*
_output_shapes
::ь¤t
*model_65/simple_rnn_21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,model_65/simple_rnn_21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,model_65/simple_rnn_21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:─
$model_65/simple_rnn_21/strided_sliceStridedSlice%model_65/simple_rnn_21/Shape:output:03model_65/simple_rnn_21/strided_slice/stack:output:05model_65/simple_rnn_21/strided_slice/stack_1:output:05model_65/simple_rnn_21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2model_65/simple_rnn_21/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         э
$model_65/simple_rnn_21/TensorArrayV2TensorListReserve;model_65/simple_rnn_21/TensorArrayV2/element_shape:output:0-model_65/simple_rnn_21/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмЮ
Lmodel_65/simple_rnn_21/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ц
>model_65/simple_rnn_21/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$model_65/simple_rnn_21/transpose:y:0Umodel_65/simple_rnn_21/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмv
,model_65/simple_rnn_21/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model_65/simple_rnn_21/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model_65/simple_rnn_21/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▄
&model_65/simple_rnn_21/strided_slice_1StridedSlice$model_65/simple_rnn_21/transpose:y:05model_65/simple_rnn_21/strided_slice_1/stack:output:07model_65/simple_rnn_21/strided_slice_1/stack_1:output:07model_65/simple_rnn_21/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask├
<model_65/simple_rnn_21/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOpEmodel_65_simple_rnn_21_simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0р
-model_65/simple_rnn_21/simple_rnn_cell/MatMulMatMul/model_65/simple_rnn_21/strided_slice_1:output:0Dmodel_65/simple_rnn_21/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ┴
=model_65/simple_rnn_21/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpFmodel_65_simple_rnn_21_simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0В
.model_65/simple_rnn_21/simple_rnn_cell/BiasAddBiasAdd7model_65/simple_rnn_21/simple_rnn_cell/MatMul:product:0Emodel_65/simple_rnn_21/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ╚
>model_65/simple_rnn_21/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpGmodel_65_simple_rnn_21_simple_rnn_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0Й
/model_65/simple_rnn_21/simple_rnn_cell/MatMul_1MatMulinput_53Fmodel_65/simple_rnn_21/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ┌
*model_65/simple_rnn_21/simple_rnn_cell/addAddV27model_65/simple_rnn_21/simple_rnn_cell/BiasAdd:output:09model_65/simple_rnn_21/simple_rnn_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђќ
+model_65/simple_rnn_21/simple_rnn_cell/TanhTanh.model_65/simple_rnn_21/simple_rnn_cell/add:z:0*
T0*(
_output_shapes
:         ђЁ
4model_65/simple_rnn_21/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   ч
&model_65/simple_rnn_21/TensorArrayV2_1TensorListReserve=model_65/simple_rnn_21/TensorArrayV2_1/element_shape:output:0-model_65/simple_rnn_21/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм]
model_65/simple_rnn_21/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/model_65/simple_rnn_21/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         k
)model_65/simple_rnn_21/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┼
model_65/simple_rnn_21/whileWhile2model_65/simple_rnn_21/while/loop_counter:output:08model_65/simple_rnn_21/while/maximum_iterations:output:0$model_65/simple_rnn_21/time:output:0/model_65/simple_rnn_21/TensorArrayV2_1:handle:0input_53-model_65/simple_rnn_21/strided_slice:output:0Nmodel_65/simple_rnn_21/TensorArrayUnstack/TensorListFromTensor:output_handle:0Emodel_65_simple_rnn_21_simple_rnn_cell_matmul_readvariableop_resourceFmodel_65_simple_rnn_21_simple_rnn_cell_biasadd_readvariableop_resourceGmodel_65_simple_rnn_21_simple_rnn_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ђ: : : : : *%
_read_only_resource_inputs
	*5
body-R+
)model_65_simple_rnn_21_while_body_7845778*5
cond-R+
)model_65_simple_rnn_21_while_cond_7845777*9
output_shapes(
&: : : : :         ђ: : : : : *
parallel_iterations ў
Gmodel_65/simple_rnn_21/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   Љ
9model_65/simple_rnn_21/TensorArrayV2Stack/TensorListStackTensorListStack%model_65/simple_rnn_21/while:output:3Pmodel_65/simple_rnn_21/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ђ*
element_dtype0
,model_65/simple_rnn_21/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         x
.model_65/simple_rnn_21/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.model_65/simple_rnn_21/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
&model_65/simple_rnn_21/strided_slice_2StridedSliceBmodel_65/simple_rnn_21/TensorArrayV2Stack/TensorListStack:tensor:05model_65/simple_rnn_21/strided_slice_2/stack:output:07model_65/simple_rnn_21/strided_slice_2/stack_1:output:07model_65/simple_rnn_21/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_mask|
'model_65/simple_rnn_21/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          т
"model_65/simple_rnn_21/transpose_1	TransposeBmodel_65/simple_rnn_21/TensorArrayV2Stack/TensorListStack:tensor:00model_65/simple_rnn_21/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  ђЪ
*model_65/dense_14/Tensordot/ReadVariableOpReadVariableOp3model_65_dense_14_tensordot_readvariableop_resource*
_output_shapes
:	ђ*
dtype0j
 model_65/dense_14/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:q
 model_65/dense_14/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Ё
!model_65/dense_14/Tensordot/ShapeShape&model_65/simple_rnn_21/transpose_1:y:0*
T0*
_output_shapes
::ь¤k
)model_65/dense_14/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ѓ
$model_65/dense_14/Tensordot/GatherV2GatherV2*model_65/dense_14/Tensordot/Shape:output:0)model_65/dense_14/Tensordot/free:output:02model_65/dense_14/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
+model_65/dense_14/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Є
&model_65/dense_14/Tensordot/GatherV2_1GatherV2*model_65/dense_14/Tensordot/Shape:output:0)model_65/dense_14/Tensordot/axes:output:04model_65/dense_14/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
!model_65/dense_14/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ц
 model_65/dense_14/Tensordot/ProdProd-model_65/dense_14/Tensordot/GatherV2:output:0*model_65/dense_14/Tensordot/Const:output:0*
T0*
_output_shapes
: m
#model_65/dense_14/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ф
"model_65/dense_14/Tensordot/Prod_1Prod/model_65/dense_14/Tensordot/GatherV2_1:output:0,model_65/dense_14/Tensordot/Const_1:output:0*
T0*
_output_shapes
: i
'model_65/dense_14/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : С
"model_65/dense_14/Tensordot/concatConcatV2)model_65/dense_14/Tensordot/free:output:0)model_65/dense_14/Tensordot/axes:output:00model_65/dense_14/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:»
!model_65/dense_14/Tensordot/stackPack)model_65/dense_14/Tensordot/Prod:output:0+model_65/dense_14/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:К
%model_65/dense_14/Tensordot/transpose	Transpose&model_65/simple_rnn_21/transpose_1:y:0+model_65/dense_14/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  ђ└
#model_65/dense_14/Tensordot/ReshapeReshape)model_65/dense_14/Tensordot/transpose:y:0*model_65/dense_14/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  └
"model_65/dense_14/Tensordot/MatMulMatMul,model_65/dense_14/Tensordot/Reshape:output:02model_65/dense_14/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         m
#model_65/dense_14/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:k
)model_65/dense_14/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : №
$model_65/dense_14/Tensordot/concat_1ConcatV2-model_65/dense_14/Tensordot/GatherV2:output:0,model_65/dense_14/Tensordot/Const_2:output:02model_65/dense_14/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:┬
model_65/dense_14/TensordotReshape,model_65/dense_14/Tensordot/MatMul:product:0-model_65/dense_14/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  ќ
(model_65/dense_14/BiasAdd/ReadVariableOpReadVariableOp1model_65_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╗
model_65/dense_14/BiasAddBiasAdd$model_65/dense_14/Tensordot:output:00model_65/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  Є
model_65/dense_14/SoftmaxSoftmax"model_65/dense_14/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  
IdentityIdentity#model_65/dense_14/Softmax:softmax:0^NoOp*
T0*4
_output_shapes"
 :                  w

Identity_1Identity%model_65/simple_rnn_21/while:output:4^NoOp*
T0*(
_output_shapes
:         ђ┘
NoOpNoOp)^model_65/dense_14/BiasAdd/ReadVariableOp+^model_65/dense_14/Tensordot/ReadVariableOp>^model_65/simple_rnn_21/simple_rnn_cell/BiasAdd/ReadVariableOp=^model_65/simple_rnn_21/simple_rnn_cell/MatMul/ReadVariableOp?^model_65/simple_rnn_21/simple_rnn_cell/MatMul_1/ReadVariableOp^model_65/simple_rnn_21/while*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:                  :         ђ: : : : : 2T
(model_65/dense_14/BiasAdd/ReadVariableOp(model_65/dense_14/BiasAdd/ReadVariableOp2X
*model_65/dense_14/Tensordot/ReadVariableOp*model_65/dense_14/Tensordot/ReadVariableOp2~
=model_65/simple_rnn_21/simple_rnn_cell/BiasAdd/ReadVariableOp=model_65/simple_rnn_21/simple_rnn_cell/BiasAdd/ReadVariableOp2|
<model_65/simple_rnn_21/simple_rnn_cell/MatMul/ReadVariableOp<model_65/simple_rnn_21/simple_rnn_cell/MatMul/ReadVariableOp2ђ
>model_65/simple_rnn_21/simple_rnn_cell/MatMul_1/ReadVariableOp>model_65/simple_rnn_21/simple_rnn_cell/MatMul_1/ReadVariableOp2<
model_65/simple_rnn_21/whilemodel_65/simple_rnn_21/while:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
(
_output_shapes
:         ђ
"
_user_specified_name
input_53:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
input_38
Г
§
E__inference_dense_14_layer_call_and_return_conditional_losses_7846309

inputs4
!tensordot_readvariableop_resource:	ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	ђ*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::ь¤Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ю
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ѓ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:                  ђі
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  і
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ї
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ё
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  c
SoftmaxSoftmaxBiasAdd:output:0*
T0*4
_output_shapes"
 :                  m
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*4
_output_shapes"
 :                  V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs
Ї#
м
while_body_7845928
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_simple_rnn_cell_7845950_0:	ђ.
while_simple_rnn_cell_7845952_0:	ђ3
while_simple_rnn_cell_7845954_0:
ђђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_7845950:	ђ,
while_simple_rnn_cell_7845952:	ђ1
while_simple_rnn_cell_7845954:
ђђѕб-while/simple_rnn_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ъ
-while/simple_rnn_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_7845950_0while_simple_rnn_cell_7845952_0while_simple_rnn_cell_7845954_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         ђ:         ђ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_7845915▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder6while/simple_rnn_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ћ
while/Identity_4Identity6while/simple_rnn_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         ђX

while/NoOpNoOp.^while/simple_rnn_cell/StatefulPartitionedCall*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"@
while_simple_rnn_cell_7845950while_simple_rnn_cell_7845950_0"@
while_simple_rnn_cell_7845952while_simple_rnn_cell_7845952_0"@
while_simple_rnn_cell_7845954while_simple_rnn_cell_7845954_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ђ: : : : : 2^
-while/simple_rnn_cell/StatefulPartitionedCall-while/simple_rnn_cell/StatefulPartitionedCall:'	#
!
_user_specified_name	7845954:'#
!
_user_specified_name	7845952:'#
!
_user_specified_name	7845950:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
З6
Н
 __inference__traced_save_7847159
file_prefix9
&read_disablecopyonread_dense_14_kernel:	ђ4
&read_1_disablecopyonread_dense_14_bias:P
=read_2_disablecopyonread_simple_rnn_21_simple_rnn_cell_kernel:	ђ[
Gread_3_disablecopyonread_simple_rnn_21_simple_rnn_cell_recurrent_kernel:
ђђJ
;read_4_disablecopyonread_simple_rnn_21_simple_rnn_cell_bias:	ђ
savev2_const
identity_11ѕбMergeV2CheckpointsбRead/DisableCopyOnReadбRead/ReadVariableOpбRead_1/DisableCopyOnReadбRead_1/ReadVariableOpбRead_2/DisableCopyOnReadбRead_2/ReadVariableOpбRead_3/DisableCopyOnReadбRead_3/ReadVariableOpбRead_4/DisableCopyOnReadбRead_4/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partЂ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_14_kernel"/device:CPU:0*
_output_shapes
 Б
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_14_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	ђ*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	ђb

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	ђz
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_14_bias"/device:CPU:0*
_output_shapes
 б
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_14_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:Љ
Read_2/DisableCopyOnReadDisableCopyOnRead=read_2_disablecopyonread_simple_rnn_21_simple_rnn_cell_kernel"/device:CPU:0*
_output_shapes
 Й
Read_2/ReadVariableOpReadVariableOp=read_2_disablecopyonread_simple_rnn_21_simple_rnn_cell_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	ђ*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	ђd

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	ђЏ
Read_3/DisableCopyOnReadDisableCopyOnReadGread_3_disablecopyonread_simple_rnn_21_simple_rnn_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ╔
Read_3/ReadVariableOpReadVariableOpGread_3_disablecopyonread_simple_rnn_21_simple_rnn_cell_recurrent_kernel^Read_3/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
ђђ*
dtype0o

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ђђe

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ђђЈ
Read_4/DisableCopyOnReadDisableCopyOnRead;read_4_disablecopyonread_simple_rnn_21_simple_rnn_cell_bias"/device:CPU:0*
_output_shapes
 И
Read_4/ReadVariableOpReadVariableOp;read_4_disablecopyonread_simple_rnn_21_simple_rnn_cell_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0j

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђ`

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђ­
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ў
valueЈBїB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHy
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B к
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes

2љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_10Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_11IdentityIdentity_10:output:0^NoOp*
T0*
_output_shapes
: ▓
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp*
_output_shapes
 "#
identity_11Identity_11:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:B>
<
_user_specified_name$"simple_rnn_21/simple_rnn_cell/bias:NJ
H
_user_specified_name0.simple_rnn_21/simple_rnn_cell/recurrent_kernel:D@
>
_user_specified_name&$simple_rnn_21/simple_rnn_cell/kernel:-)
'
_user_specified_namedense_14/bias:/+
)
_user_specified_namedense_14/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Љ.
┐
while_body_7846734
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0I
6while_simple_rnn_cell_matmul_readvariableop_resource_0:	ђF
7while_simple_rnn_cell_biasadd_readvariableop_resource_0:	ђL
8while_simple_rnn_cell_matmul_1_readvariableop_resource_0:
ђђ
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorG
4while_simple_rnn_cell_matmul_readvariableop_resource:	ђD
5while_simple_rnn_cell_biasadd_readvariableop_resource:	ђJ
6while_simple_rnn_cell_matmul_1_readvariableop_resource:
ђђѕб,while/simple_rnn_cell/BiasAdd/ReadVariableOpб+while/simple_rnn_cell/MatMul/ReadVariableOpб-while/simple_rnn_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Б
+while/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp6while_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes
:	ђ*
dtype0└
while/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:03while/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђА
,while/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp7while_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:ђ*
dtype0╣
while/simple_rnn_cell/BiasAddBiasAdd&while/simple_rnn_cell/MatMul:product:04while/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђе
-while/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp8while_simple_rnn_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ђђ*
dtype0Д
while/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_25while/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђД
while/simple_rnn_cell/addAddV2&while/simple_rnn_cell/BiasAdd:output:0(while/simple_rnn_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђt
while/simple_rnn_cell/TanhTanhwhile/simple_rnn_cell/add:z:0*
T0*(
_output_shapes
:         ђК
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/simple_rnn_cell/Tanh:y:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: |
while/Identity_4Identitywhile/simple_rnn_cell/Tanh:y:0^while/NoOp*
T0*(
_output_shapes
:         ђх

while/NoOpNoOp-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"p
5while_simple_rnn_cell_biasadd_readvariableop_resource7while_simple_rnn_cell_biasadd_readvariableop_resource_0"r
6while_simple_rnn_cell_matmul_1_readvariableop_resource8while_simple_rnn_cell_matmul_1_readvariableop_resource_0"n
4while_simple_rnn_cell_matmul_readvariableop_resource6while_simple_rnn_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ђ: : : : : 2\
,while/simple_rnn_cell/BiasAdd/ReadVariableOp,while/simple_rnn_cell/BiasAdd/ReadVariableOp2Z
+while/simple_rnn_cell/MatMul/ReadVariableOp+while/simple_rnn_cell/MatMul/ReadVariableOp2^
-while/simple_rnn_cell/MatMul_1/ReadVariableOp-while/simple_rnn_cell/MatMul_1/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
У
Г
while_cond_7846202
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_7846202___redundant_placeholder05
1while_while_cond_7846202___redundant_placeholder15
1while_while_cond_7846202___redundant_placeholder25
1while_while_cond_7846202___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ђ: :::::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
┤8
п
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7846270

inputs
initial_stateA
.simple_rnn_cell_matmul_readvariableop_resource:	ђ>
/simple_rnn_cell_biasadd_readvariableop_resource:	ђD
0simple_rnn_cell_matmul_1_readvariableop_resource:
ђђ
identity

identity_1ѕб&simple_rnn_cell/BiasAdd/ReadVariableOpб%simple_rnn_cell/MatMul/ReadVariableOpб'simple_rnn_cell/MatMul_1/ReadVariableOpбwhilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  P
ShapeShapetranspose:y:0*
T0*
_output_shapes
::ь¤]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЋ
%simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp.simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0ю
simple_rnn_cell/MatMulMatMulstrided_slice_1:output:0-simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЊ
&simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp/simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Д
simple_rnn_cell/BiasAddBiasAdd simple_rnn_cell/MatMul:product:0.simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђџ
'simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp0simple_rnn_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0Ћ
simple_rnn_cell/MatMul_1MatMulinitial_state/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
simple_rnn_cell/addAddV2 simple_rnn_cell/BiasAdd:output:0"simple_rnn_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ђh
simple_rnn_cell/TanhTanhsimple_rnn_cell/add:z:0*
T0*(
_output_shapes
:         ђn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Х
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0initial_statestrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0.simple_rnn_cell_matmul_readvariableop_resource/simple_rnn_cell_biasadd_readvariableop_resource0simple_rnn_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ђ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_7846203*
condR
while_cond_7846202*9
output_shapes(
&: : : : :         ђ: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   ╠
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ђ*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ђ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  ђl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  ђ`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:         ђЦ
NoOpNoOp'^simple_rnn_cell/BiasAdd/ReadVariableOp&^simple_rnn_cell/MatMul/ReadVariableOp(^simple_rnn_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::                  :         ђ: : : 2P
&simple_rnn_cell/BiasAdd/ReadVariableOp&simple_rnn_cell/BiasAdd/ReadVariableOp2N
%simple_rnn_cell/MatMul/ReadVariableOp%simple_rnn_cell/MatMul/ReadVariableOp2R
'simple_rnn_cell/MatMul_1/ReadVariableOp'simple_rnn_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:WS
(
_output_shapes
:         ђ
'
_user_specified_nameinitial_state:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
и
■
#__inference__traced_restore_7847183
file_prefix3
 assignvariableop_dense_14_kernel:	ђ.
 assignvariableop_1_dense_14_bias:J
7assignvariableop_2_simple_rnn_21_simple_rnn_cell_kernel:	ђU
Aassignvariableop_3_simple_rnn_21_simple_rnn_cell_recurrent_kernel:
ђђD
5assignvariableop_4_simple_rnn_21_simple_rnn_cell_bias:	ђ

identity_6ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4з
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ў
valueЈBїB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH|
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B ╝
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:│
AssignVariableOpAssignVariableOp assignvariableop_dense_14_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_14_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_2AssignVariableOp7assignvariableop_2_simple_rnn_21_simple_rnn_cell_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_3AssignVariableOpAassignvariableop_3_simple_rnn_21_simple_rnn_cell_recurrent_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_4AssignVariableOp5assignvariableop_4_simple_rnn_21_simple_rnn_cell_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ┴

Identity_5Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_6IdentityIdentity_5:output:0^NoOp_1*
T0*
_output_shapes
: І
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4*
_output_shapes
 "!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42$
AssignVariableOpAssignVariableOp:B>
<
_user_specified_name$"simple_rnn_21/simple_rnn_cell/bias:NJ
H
_user_specified_name0.simple_rnn_21/simple_rnn_cell/recurrent_kernel:D@
>
_user_specified_name&$simple_rnn_21/simple_rnn_cell/kernel:-)
'
_user_specified_namedense_14/bias:/+
)
_user_specified_namedense_14/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ь
»
while_cond_7846733
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_7846733___redundant_placeholder05
1while_while_cond_7846733___redundant_placeholder15
1while_while_cond_7846733___redundant_placeholder25
1while_while_cond_7846733___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ђ: :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:.*
(
_output_shapes
:         ђ:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Г
§
E__inference_dense_14_layer_call_and_return_conditional_losses_7847043

inputs4
!tensordot_readvariableop_resource:	ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	ђ*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::ь¤Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ю
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ѓ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:                  ђі
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  і
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ї
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ё
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  c
SoftmaxSoftmaxBiasAdd:output:0*
T0*4
_output_shapes"
 :                  m
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*4
_output_shapes"
 :                  V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
5
_output_shapes#
!:                  ђ
 
_user_specified_nameinputs"ДL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╦
serving_defaultи
J
input_38>
serving_default_input_38:0                  
>
input_532
serving_default_input_53:0         ђI
dense_14=
StatefulPartitionedCall:0                  B
simple_rnn_211
StatefulPartitionedCall:1         ђtensorflow/serving/predict:┬Ѕ
╝
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
├
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec"
_tf_keras_rnn_layer
╗
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
C
0
1
2
3
4"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
 non_trainable_variables

!layers
"metrics
#layer_regularization_losses
$layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
К
%trace_0
&trace_12љ
*__inference_model_65_layer_call_fn_7846455
*__inference_model_65_layer_call_fn_7846473х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z%trace_0z&trace_1
§
'trace_0
(trace_12к
E__inference_model_65_layer_call_and_return_conditional_losses_7846317
E__inference_model_65_layer_call_and_return_conditional_losses_7846437х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z'trace_0z(trace_1
пBН
"__inference__wrapped_model_7845872input_38input_53"ў
Љ▓Ї
FullArgSpec
argsџ

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
,
)serving_default"
signature_map
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
╣

*states
+non_trainable_variables

,layers
-metrics
.layer_regularization_losses
/layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ч
0trace_0
1trace_1
2trace_2
3trace_32Љ
/__inference_simple_rnn_21_layer_call_fn_7846542
/__inference_simple_rnn_21_layer_call_fn_7846555
/__inference_simple_rnn_21_layer_call_fn_7846569
/__inference_simple_rnn_21_layer_call_fn_7846583╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z0trace_0z1trace_1z2trace_2z3trace_3
У
4trace_0
5trace_1
6trace_2
7trace_32§
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7846692
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7846801
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7846902
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7847003╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z4trace_0z5trace_1z6trace_2z7trace_3
У
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
>_random_generator

kernel
recurrent_kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
С
Dtrace_02К
*__inference_dense_14_layer_call_fn_7847012ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zDtrace_0
 
Etrace_02Р
E__inference_dense_14_layer_call_and_return_conditional_losses_7847043ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zEtrace_0
": 	ђ2dense_14/kernel
:2dense_14/bias
7:5	ђ2$simple_rnn_21/simple_rnn_cell/kernel
B:@
ђђ2.simple_rnn_21/simple_rnn_cell/recurrent_kernel
1:/ђ2"simple_rnn_21/simple_rnn_cell/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЗBы
*__inference_model_65_layer_call_fn_7846455input_38input_53"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЗBы
*__inference_model_65_layer_call_fn_7846473input_38input_53"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЈBї
E__inference_model_65_layer_call_and_return_conditional_losses_7846317input_38input_53"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЈBї
E__inference_model_65_layer_call_and_return_conditional_losses_7846437input_38input_53"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
уBС
%__inference_signature_wrapper_7846529input_38input_53"д
Ъ▓Џ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 )

kwonlyargsџ

jinput_38

jinput_53
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ђB§
/__inference_simple_rnn_21_layer_call_fn_7846542inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
/__inference_simple_rnn_21_layer_call_fn_7846555inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЈBї
/__inference_simple_rnn_21_layer_call_fn_7846569inputsinitial_state_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЈBї
/__inference_simple_rnn_21_layer_call_fn_7846583inputsinitial_state_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЏBў
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7846692inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЏBў
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7846801inputs_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
фBД
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7846902inputsinitial_state_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
фBД
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7847003inputsinitial_state_0"й
Х▓▓
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
М
Ktrace_0
Ltrace_12ю
1__inference_simple_rnn_cell_layer_call_fn_7847057
1__inference_simple_rnn_cell_layer_call_fn_7847071│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zKtrace_0zLtrace_1
Ѕ
Mtrace_0
Ntrace_12м
L__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_7847088
L__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_7847105│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zMtrace_0zNtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
нBЛ
*__inference_dense_14_layer_call_fn_7847012inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№BВ
E__inference_dense_14_layer_call_and_return_conditional_losses_7847043inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
чBЭ
1__inference_simple_rnn_cell_layer_call_fn_7847057inputsstates_0"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
чBЭ
1__inference_simple_rnn_cell_layer_call_fn_7847071inputsstates_0"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ќBЊ
L__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_7847088inputsstates_0"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ќBЊ
L__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_7847105inputsstates_0"«
Д▓Б
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 Ћ
"__inference__wrapped_model_7845872Ьhбe
^б[
YџV
/і,
input_38                  
#і 
input_53         ђ
ф "{фx
;
dense_14/і,
dense_14                  
9
simple_rnn_21(і%
simple_rnn_21         ђК
E__inference_dense_14_layer_call_and_return_conditional_losses_7847043~=б:
3б0
.і+
inputs                  ђ
ф "9б6
/і,
tensor_0                  
џ А
*__inference_dense_14_layer_call_fn_7847012s=б:
3б0
.і+
inputs                  ђ
ф ".і+
unknown                  г
E__inference_model_65_layer_call_and_return_conditional_losses_7846317Рpбm
fбc
YџV
/і,
input_38                  
#і 
input_53         ђ
p

 
ф "gбd
]џZ
1і.

tensor_0_0                  
%і"

tensor_0_1         ђ
џ г
E__inference_model_65_layer_call_and_return_conditional_losses_7846437Рpбm
fбc
YџV
/і,
input_38                  
#і 
input_53         ђ
p 

 
ф "gбd
]џZ
1і.

tensor_0_0                  
%і"

tensor_0_1         ђ
џ Ѓ
*__inference_model_65_layer_call_fn_7846455нpбm
fбc
YџV
/і,
input_38                  
#і 
input_53         ђ
p

 
ф "YџV
/і,
tensor_0                  
#і 
tensor_1         ђЃ
*__inference_model_65_layer_call_fn_7846473нpбm
fбc
YџV
/і,
input_38                  
#і 
input_53         ђ
p 

 
ф "YџV
/і,
tensor_0                  
#і 
tensor_1         ђФ
%__inference_signature_wrapper_7846529Ђ{бx
б 
qфn
;
input_38/і,
input_38                  
/
input_53#і 
input_53         ђ"{фx
;
dense_14/і,
dense_14                  
9
simple_rnn_21(і%
simple_rnn_21         ђЈ
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7846692└OбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф "hбe
^џ[
2і/

tensor_0_0                  ђ
%і"

tensor_0_1         ђ
џ Ј
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7846801└OбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф "hбe
^џ[
2і/

tensor_0_0                  ђ
%і"

tensor_0_1         ђ
џ х
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7846902Тuбr
kбh
-і*
inputs                  

 
p
/џ,
*і'
initial_state_0         ђ
ф "hбe
^џ[
2і/

tensor_0_0                  ђ
%і"

tensor_0_1         ђ
џ х
J__inference_simple_rnn_21_layer_call_and_return_conditional_losses_7847003Тuбr
kбh
-і*
inputs                  

 
p 
/џ,
*і'
initial_state_0         ђ
ф "hбe
^џ[
2і/

tensor_0_0                  ђ
%і"

tensor_0_1         ђ
џ Т
/__inference_simple_rnn_21_layer_call_fn_7846542▓OбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф "ZџW
0і-
tensor_0                  ђ
#і 
tensor_1         ђТ
/__inference_simple_rnn_21_layer_call_fn_7846555▓OбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф "ZџW
0і-
tensor_0                  ђ
#і 
tensor_1         ђї
/__inference_simple_rnn_21_layer_call_fn_7846569пuбr
kбh
-і*
inputs                  

 
p
/џ,
*і'
initial_state_0         ђ
ф "ZџW
0і-
tensor_0                  ђ
#і 
tensor_1         ђї
/__inference_simple_rnn_21_layer_call_fn_7846583пuбr
kбh
-і*
inputs                  

 
p 
/џ,
*і'
initial_state_0         ђ
ф "ZџW
0і-
tensor_0                  ђ
#і 
tensor_1         ђЎ
L__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_7847088╚]бZ
SбP
 і
inputs         
(б%
#і 
states_0         ђ
p
ф "bб_
XбU
%і"

tensor_0_0         ђ
,џ)
'і$
tensor_0_1_0         ђ
џ Ў
L__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_7847105╚]бZ
SбP
 і
inputs         
(б%
#і 
states_0         ђ
p 
ф "bб_
XбU
%і"

tensor_0_0         ђ
,џ)
'і$
tensor_0_1_0         ђ
џ ­
1__inference_simple_rnn_cell_layer_call_fn_7847057║]бZ
SбP
 і
inputs         
(б%
#і 
states_0         ђ
p
ф "TбQ
#і 
tensor_0         ђ
*џ'
%і"

tensor_1_0         ђ­
1__inference_simple_rnn_cell_layer_call_fn_7847071║]бZ
SбP
 і
inputs         
(б%
#і 
states_0         ђ
p 
ф "TбQ
#і 
tensor_0         ђ
*џ'
%і"

tensor_1_0         ђ