
?
:main_level/agent/main/online/global_step/Initializer/zerosConst*;
_class1
/-loc:@main_level/agent/main/online/global_step*
value	B	 R *
dtype0	
?
(main_level/agent/main/online/global_step
VariableV2"/device:GPU:0*
shared_name *;
_class1
/-loc:@main_level/agent/main/online/global_step*
dtype0	*
	container *
shape: 
?
/main_level/agent/main/online/global_step/AssignAssign(main_level/agent/main/online/global_step:main_level/agent/main/online/global_step/Initializer/zeros"/device:GPU:0*
validate_shape(*
use_locking(*
T0	*;
_class1
/-loc:@main_level/agent/main/online/global_step
?
-main_level/agent/main/online/global_step/readIdentity(main_level/agent/main/online/global_step"/device:GPU:0*
T0	*;
_class1
/-loc:@main_level/agent/main/online/global_step
l
3main_level/agent/main/online/Variable/initial_valueConst"/device:GPU:0*
value	B
 Z *
dtype0

?
%main_level/agent/main/online/Variable
VariableV2"/device:GPU:0*
shape: *
shared_name *
dtype0
*
	container 
?
,main_level/agent/main/online/Variable/AssignAssign%main_level/agent/main/online/Variable3main_level/agent/main/online/Variable/initial_value"/device:GPU:0*
T0
*8
_class.
,*loc:@main_level/agent/main/online/Variable*
validate_shape(*
use_locking(
?
*main_level/agent/main/online/Variable/readIdentity%main_level/agent/main/online/Variable"/device:GPU:0*
T0
*8
_class.
,*loc:@main_level/agent/main/online/Variable
b
(main_level/agent/main/online/PlaceholderPlaceholder"/device:GPU:0*
shape:*
dtype0

?
#main_level/agent/main/online/AssignAssign%main_level/agent/main/online/Variable(main_level/agent/main/online/Placeholder"/device:GPU:0*
use_locking(*
T0
*8
_class.
,*loc:@main_level/agent/main/online/Variable*
validate_shape(
?
>main_level/agent/main/online/network_0/observation/observationPlaceholder"/device:GPU:0*
dtype0*
shape:??????????
x
<main_level/agent/main/online/network_0/observation/truediv/yConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
:main_level/agent/main/online/network_0/observation/truedivRealDiv>main_level/agent/main/online/network_0/observation/observation<main_level/agent/main/online/network_0/observation/truediv/y"/device:GPU:0*
T0
t
8main_level/agent/main/online/network_0/observation/sub/yConst"/device:GPU:0*
valueB
 *    *
dtype0
?
6main_level/agent/main/online/network_0/observation/subSub:main_level/agent/main/online/network_0/observation/truediv8main_level/agent/main/online/network_0/observation/sub/y"/device:GPU:0*
T0
?
omain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/shapeConst*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean*
valueB"H     *
dtype0
?
mmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/minConst*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean*
valueB
 *OS?*
dtype0
?
mmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/maxConst*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean*
valueB
 *OS=*
dtype0
?
wmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/RandomUniformRandomUniformomain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/shape*

seed *
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean*
dtype0*
seed2 
?
mmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/subSubmmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/maxmmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/min*
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean
?
mmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/mulMulwmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/RandomUniformmmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/sub*
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean
?
imain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniformAddmmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/mulmmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/min*
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean
?
Nmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean
VariableV2"/device:GPU:0*
dtype0*
	container *
shape:
??*
shared_name *a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean
?
Umain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/AssignAssignNmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_meanimain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean*
validate_shape(
?
Smain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/readIdentityNmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean"/device:GPU:0*
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean
?
^main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Initializer/zerosConst*
dtype0*_
_classU
SQloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean*
valueB?*    
?
Lmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean
VariableV2"/device:GPU:0*
shape:?*
shared_name *_
_classU
SQloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean*
dtype0*
	container 
?
Smain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/AssignAssignLmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean^main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean*
validate_shape(
?
Qmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/readIdentityLmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean"/device:GPU:0*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean
?
qmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/shapeConst*c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev*
valueB"H     *
dtype0
?
omain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/minConst*
dtype0*c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev*
valueB
 *OS??
?
omain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/maxConst*c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev*
valueB
 *OS?<*
dtype0
?
ymain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/RandomUniformRandomUniformqmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/shape*

seed *
T0*c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev*
dtype0*
seed2 
?
omain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/subSubomain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/maxomain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/min*
T0*c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev
?
omain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/mulMulymain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/RandomUniformomain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/sub*
T0*c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev
?
kmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniformAddomain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/mulomain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/min*
T0*c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev
?
Pmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev
VariableV2"/device:GPU:0*c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev*
dtype0*
	container *
shape:
??*
shared_name 
?
Wmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/AssignAssignPmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddevkmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform"/device:GPU:0*
validate_shape(*
use_locking(*
T0*c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev
?
Umain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/readIdentityPmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev"/device:GPU:0*
T0*c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev
?
omain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/shapeConst*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev*
valueB:?*
dtype0
?
mmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/minConst*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev*
valueB
 *OS??*
dtype0
?
mmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/maxConst*
dtype0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev*
valueB
 *OS?<
?
wmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/RandomUniformRandomUniformomain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/shape*
seed2 *

seed *
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev*
dtype0
?
mmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/subSubmmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/maxmmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/min*
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev
?
mmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/mulMulwmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/RandomUniformmmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/sub*
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev
?
imain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniformAddmmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/mulmmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/min*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev*
T0
?
Nmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev
VariableV2"/device:GPU:0*
shape:?*
shared_name *a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev*
dtype0*
	container 
?
Umain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/AssignAssignNmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddevimain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform"/device:GPU:0*
validate_shape(*
use_locking(*
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev
?
Smain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/readIdentityNmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev"/device:GPU:0*
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev
?
Vmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal/shapeConst"/device:GPU:0*
dtype0*
valueB:?
?
Umain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal/meanConst"/device:GPU:0*
dtype0*
valueB
 *    
?
Wmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
emain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal/RandomStandardNormalRandomStandardNormalVmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal/shape"/device:GPU:0*
T0*
dtype0*
seed2 *

seed 
?
Tmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal/mulMulemain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal/RandomStandardNormalWmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal/stddev"/device:GPU:0*
T0
?
Pmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normalAddTmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal/mulUmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal/mean"/device:GPU:0*
T0
?
Fmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/AbsAbsPmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal"/device:GPU:0*
T0
?
Gmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/SqrtSqrtFmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/Abs"/device:GPU:0*
T0
?
Gmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/SignSignPmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal"/device:GPU:0*
T0
?
Fmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/mulMulGmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/SqrtGmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/Sign"/device:GPU:0*
T0
?
Xmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_1/shapeConst"/device:GPU:0*
valueB"H     *
dtype0
?
Wmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_1/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
Ymain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_1/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
gmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_1/RandomStandardNormalRandomStandardNormalXmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_1/shape"/device:GPU:0*
seed2 *

seed *
T0*
dtype0
?
Vmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_1/mulMulgmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_1/RandomStandardNormalYmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_1/stddev"/device:GPU:0*
T0
?
Rmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_1AddVmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_1/mulWmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_1/mean"/device:GPU:0*
T0
?
Hmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/Abs_1AbsRmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_1"/device:GPU:0*
T0
?
Imain_level/agent/main/online/network_0/observation/NoisyNetDense_0/Sqrt_1SqrtHmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/Abs_1"/device:GPU:0*
T0
?
Imain_level/agent/main/online/network_0/observation/NoisyNetDense_0/Sign_1SignRmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_1"/device:GPU:0*
T0
?
Hmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/mul_1MulImain_level/agent/main/online/network_0/observation/NoisyNetDense_0/Sqrt_1Imain_level/agent/main/online/network_0/observation/NoisyNetDense_0/Sign_1"/device:GPU:0*
T0
?
Xmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_2/shapeConst"/device:GPU:0*
valueB"      *
dtype0
?
Wmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_2/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
Ymain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_2/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
gmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_2/RandomStandardNormalRandomStandardNormalXmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_2/shape"/device:GPU:0*
T0*
dtype0*
seed2 *

seed 
?
Vmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_2/mulMulgmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_2/RandomStandardNormalYmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_2/stddev"/device:GPU:0*
T0
?
Rmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_2AddVmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_2/mulWmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_2/mean"/device:GPU:0*
T0
?
Hmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/Abs_2AbsRmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_2"/device:GPU:0*
T0
?
Imain_level/agent/main/online/network_0/observation/NoisyNetDense_0/Sqrt_2SqrtHmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/Abs_2"/device:GPU:0*
T0
?
Imain_level/agent/main/online/network_0/observation/NoisyNetDense_0/Sign_2SignRmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/random_normal_2"/device:GPU:0*
T0
?
Hmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/mul_2MulImain_level/agent/main/online/network_0/observation/NoisyNetDense_0/Sqrt_2Imain_level/agent/main/online/network_0/observation/NoisyNetDense_0/Sign_2"/device:GPU:0*
T0
?
Imain_level/agent/main/online/network_0/observation/NoisyNetDense_0/MatMulMatMulHmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/mul_1Hmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/mul_2"/device:GPU:0*
T0*
transpose_a( *
transpose_b( 
?
6main_level/agent/main/online/network_0/observation/mulMulSmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/readFmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/mul"/device:GPU:0*
T0
?
6main_level/agent/main/online/network_0/observation/addAddQmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/read6main_level/agent/main/online/network_0/observation/mul"/device:GPU:0*
T0
?
8main_level/agent/main/online/network_0/observation/mul_1MulUmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/readImain_level/agent/main/online/network_0/observation/NoisyNetDense_0/MatMul"/device:GPU:0*
T0
?
8main_level/agent/main/online/network_0/observation/add_1AddSmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/read8main_level/agent/main/online/network_0/observation/mul_1"/device:GPU:0*
T0
?
9main_level/agent/main/online/network_0/observation/MatMulMatMul6main_level/agent/main/online/network_0/observation/sub8main_level/agent/main/online/network_0/observation/add_1"/device:GPU:0*
T0*
transpose_a( *
transpose_b( 
?
8main_level/agent/main/online/network_0/observation/add_2Add9main_level/agent/main/online/network_0/observation/MatMul6main_level/agent/main/online/network_0/observation/add"/device:GPU:0*
T0
?
Zmain_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activationRelu8main_level/agent/main/online/network_0/observation/add_2"/device:GPU:0*
T0
?
Hmain_level/agent/main/online/network_0/observation/Flatten/flatten/ShapeShapeZmain_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0
?
Vmain_level/agent/main/online/network_0/observation/Flatten/flatten/strided_slice/stackConst"/device:GPU:0*
valueB: *
dtype0
?
Xmain_level/agent/main/online/network_0/observation/Flatten/flatten/strided_slice/stack_1Const"/device:GPU:0*
dtype0*
valueB:
?
Xmain_level/agent/main/online/network_0/observation/Flatten/flatten/strided_slice/stack_2Const"/device:GPU:0*
valueB:*
dtype0
?
Pmain_level/agent/main/online/network_0/observation/Flatten/flatten/strided_sliceStridedSliceHmain_level/agent/main/online/network_0/observation/Flatten/flatten/ShapeVmain_level/agent/main/online/network_0/observation/Flatten/flatten/strided_slice/stackXmain_level/agent/main/online/network_0/observation/Flatten/flatten/strided_slice/stack_1Xmain_level/agent/main/online/network_0/observation/Flatten/flatten/strided_slice/stack_2"/device:GPU:0*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
?
Rmain_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape/shape/1Const"/device:GPU:0*
dtype0*
valueB :
?????????
?
Pmain_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape/shapePackPmain_level/agent/main/online/network_0/observation/Flatten/flatten/strided_sliceRmain_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape/shape/1"/device:GPU:0*
T0*

axis *
N
?
Jmain_level/agent/main/online/network_0/observation/Flatten/flatten/ReshapeReshapeZmain_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activationPmain_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape/shape"/device:GPU:0*
Tshape0*
T0
}
=main_level/agent/main/online/network_0/Variable/initial_valueConst"/device:GPU:0*
valueB*  ??*
dtype0
?
/main_level/agent/main/online/network_0/Variable
VariableV2"/device:GPU:0*
dtype0*
	container *
shape:*
shared_name 
?
6main_level/agent/main/online/network_0/Variable/AssignAssign/main_level/agent/main/online/network_0/Variable=main_level/agent/main/online/network_0/Variable/initial_value"/device:GPU:0*
validate_shape(*
use_locking(*
T0*B
_class8
64loc:@main_level/agent/main/online/network_0/Variable
?
4main_level/agent/main/online/network_0/Variable/readIdentity/main_level/agent/main/online/network_0/Variable"/device:GPU:0*
T0*B
_class8
64loc:@main_level/agent/main/online/network_0/Variable
?
,main_level/agent/main/online/network_0/ConstConst"/device:GPU:0*?
value?B?3"?   ????33????ff?   ?33??ff??????????  ??33??ff???????̌?  ??fff???L?333????   ???̿??????L???̾    ???>??L?????????   @??@333@??L@fff@  ?@?̌@???@ff?@33?@  ?@???@???@ff?@33?@   AffA??A33A??A   A*
dtype0
?
+main_level/agent/main/online/network_0/CastCast,main_level/agent/main/online/network_0/Const"/device:GPU:0*

SrcT0*
Truncate( *

DstT0
A
ConstConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
Vmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/initial_valueConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
Hmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers
VariableV2"/device:GPU:0*
dtype0*
	container *
shape: *
shared_name 
?
Omain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/AssignAssignHmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalersVmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/initial_value"/device:GPU:0*
use_locking(*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
validate_shape(
?
Mmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/readIdentityHmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers"/device:GPU:0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
T0
?
Jmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers_1Placeholder"/device:GPU:0*
dtype0*
shape:
?
-main_level/agent/main/online/network_0/AssignAssignHmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalersJmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers_1"/device:GPU:0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
validate_shape(*
use_locking( *
T0
h
,main_level/agent/main/online/network_0/sub/xConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
*main_level/agent/main/online/network_0/subSub,main_level/agent/main/online/network_0/sub/xMmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/read"/device:GPU:0*
T0
?
9main_level/agent/main/online/network_0/StopGradient/inputPackJmain_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape"/device:GPU:0*
T0*

axis *
N
?
3main_level/agent/main/online/network_0/StopGradientStopGradient9main_level/agent/main/online/network_0/StopGradient/input"/device:GPU:0*
T0
?
*main_level/agent/main/online/network_0/mulMul*main_level/agent/main/online/network_0/sub3main_level/agent/main/online/network_0/StopGradient"/device:GPU:0*
T0
?
.main_level/agent/main/online/network_0/mul_1/yPackJmain_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape"/device:GPU:0*
N*
T0*

axis 
?
,main_level/agent/main/online/network_0/mul_1MulMmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/read.main_level/agent/main/online/network_0/mul_1/y"/device:GPU:0*
T0
?
*main_level/agent/main/online/network_0/addAdd*main_level/agent/main/online/network_0/mul,main_level/agent/main/online/network_0/mul_1"/device:GPU:0*
T0
?
Rmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice/stackConst"/device:GPU:0*
dtype0*
valueB: 
?
Tmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice/stack_1Const"/device:GPU:0*
dtype0*
valueB:
?
Tmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice/stack_2Const"/device:GPU:0*
valueB:*
dtype0
?
Lmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_sliceStridedSlice*main_level/agent/main/online/network_0/addRmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice/stackTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice/stack_1Tmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice/stack_2"/device:GPU:0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0
?
{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/shapeConst*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
valueB"      *
dtype0
?
ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/minConst*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
valueB
 *  ??*
dtype0
?
ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/maxConst*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
valueB
 *  ?=*
dtype0
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/RandomUniformRandomUniform{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/shape*

seed *
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
dtype0*
seed2 
?
ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/subSubymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/maxymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/min*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean
?
ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/mulMul?main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/RandomUniformymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/sub*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean
?
umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniformAddymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/mulymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/min*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean
?
Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean
VariableV2"/device:GPU:0*
shape:
??*
shared_name *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
dtype0*
	container 
?
amain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/AssignAssignZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_meanumain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
validate_shape(
?
_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/readIdentityZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean"/device:GPU:0*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean
?
jmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Initializer/zerosConst*k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean*
valueB?*    *
dtype0
?
Xmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean
VariableV2"/device:GPU:0*
dtype0*
	container *
shape:?*
shared_name *k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean
?
_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/AssignAssignXmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_meanjmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Initializer/zeros"/device:GPU:0*
T0*k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean*
validate_shape(*
use_locking(
?
]main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/readIdentityXmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean"/device:GPU:0*
T0*k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean
?
}main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/shapeConst*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
valueB"      *
dtype0
?
{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/minConst*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
valueB
 *   ?*
dtype0
?
{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/maxConst*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
valueB
 *   =*
dtype0
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/RandomUniformRandomUniform}main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/shape*
T0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
dtype0*
seed2 *

seed 
?
{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/subSub{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/max{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/min*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
T0
?
{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/mulMul?main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/RandomUniform{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/sub*
T0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev
?
wmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniformAdd{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/mul{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/min*
T0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev
?
\main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev
VariableV2"/device:GPU:0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
dtype0*
	container *
shape:
??*
shared_name 
?
cmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/AssignAssign\main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddevwmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform"/device:GPU:0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
validate_shape(*
use_locking(*
T0
?
amain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/readIdentity\main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev"/device:GPU:0*
T0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev
?
{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/shapeConst*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev*
valueB:?*
dtype0
?
ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/minConst*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev*
valueB
 *   ?*
dtype0
?
ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/maxConst*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev*
valueB
 *   =*
dtype0
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/RandomUniformRandomUniform{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/shape*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev*
dtype0*
seed2 *

seed *
T0
?
ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/subSubymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/maxymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/min*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev
?
ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/mulMul?main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/RandomUniformymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/sub*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev*
T0
?
umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniformAddymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/mulymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/min*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev
?
Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev
VariableV2"/device:GPU:0*
shared_name *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev*
dtype0*
	container *
shape:?
?
amain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/AssignAssignZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddevumain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev*
validate_shape(
?
_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/readIdentityZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev"/device:GPU:0*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev
?
bmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal/shapeConst"/device:GPU:0*
valueB:?*
dtype0
?
amain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
cmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
qmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal/RandomStandardNormalRandomStandardNormalbmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal/shape"/device:GPU:0*
dtype0*
seed2 *

seed *
T0
?
`main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal/mulMulqmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal/RandomStandardNormalcmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal/stddev"/device:GPU:0*
T0
?
\main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normalAdd`main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal/mulamain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal/mean"/device:GPU:0*
T0
?
Rmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/AbsAbs\main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal"/device:GPU:0*
T0
?
Smain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/SqrtSqrtRmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/Abs"/device:GPU:0*
T0
?
Smain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/SignSign\main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal"/device:GPU:0*
T0
?
Rmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/mulMulSmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/SqrtSmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/Sign"/device:GPU:0*
T0
?
dmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1/shapeConst"/device:GPU:0*
valueB"      *
dtype0
?
cmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
emain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
smain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1/RandomStandardNormalRandomStandardNormaldmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1/shape"/device:GPU:0*
dtype0*
seed2 *

seed *
T0
?
bmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1/mulMulsmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1/RandomStandardNormalemain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1/stddev"/device:GPU:0*
T0
?
^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1Addbmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1/mulcmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1/mean"/device:GPU:0*
T0
?
Tmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/Abs_1Abs^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1"/device:GPU:0*
T0
?
Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/Sqrt_1SqrtTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/Abs_1"/device:GPU:0*
T0
?
Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/Sign_1Sign^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1"/device:GPU:0*
T0
?
Tmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/mul_1MulUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/Sqrt_1Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/Sign_1"/device:GPU:0*
T0
?
dmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2/shapeConst"/device:GPU:0*
valueB"      *
dtype0
?
cmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
emain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2/stddevConst"/device:GPU:0*
dtype0*
valueB
 *  ??
?
smain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2/RandomStandardNormalRandomStandardNormaldmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2/shape"/device:GPU:0*
dtype0*
seed2 *

seed *
T0
?
bmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2/mulMulsmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2/RandomStandardNormalemain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2/stddev"/device:GPU:0*
T0
?
^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2Addbmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2/mulcmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2/mean"/device:GPU:0*
T0
?
Tmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/Abs_2Abs^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2"/device:GPU:0*
T0
?
Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/Sqrt_2SqrtTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/Abs_2"/device:GPU:0*
T0
?
Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/Sign_2Sign^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2"/device:GPU:0*
T0
?
Tmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/mul_2MulUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/Sqrt_2Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/Sign_2"/device:GPU:0*
T0
?
Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/MatMulMatMulTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/mul_1Tmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/mul_2"/device:GPU:0*
transpose_a( *
transpose_b( *
T0
?
Nmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mulMul_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/readRmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/mul"/device:GPU:0*
T0
?
Nmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/addAdd]main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/readNmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul"/device:GPU:0*
T0
?
Pmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_1Mulamain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/readUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/MatMul"/device:GPU:0*
T0
?
Pmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_1Add_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/readPmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_1"/device:GPU:0*
T0
?
Qmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMulMatMulLmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slicePmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_1"/device:GPU:0*
transpose_a( *
transpose_b( *
T0
?
Pmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2AddQmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMulNmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add"/device:GPU:0*
T0
?
Omain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ReluReluPmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2"/device:GPU:0*
T0
?
{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/shapeConst*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean*
valueB"   3   *
dtype0
?
ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/minConst*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean*
valueB
 *?5?*
dtype0
?
ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/maxConst*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean*
valueB
 *?5=*
dtype0
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/RandomUniformRandomUniform{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/shape*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean*
dtype0*
seed2 *

seed *
T0
?
ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/subSubymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/maxymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/min*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean*
T0
?
ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/mulMul?main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/RandomUniformymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/sub*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean
?
umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniformAddymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/mulymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/min*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean
?
Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean
VariableV2"/device:GPU:0*
dtype0*
	container *
shape:	?3*
shared_name *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean
?
amain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/AssignAssignZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_meanumain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform"/device:GPU:0*
validate_shape(*
use_locking(*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean
?
_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/readIdentityZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean"/device:GPU:0*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean
?
jmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Initializer/zerosConst*k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean*
valueB3*    *
dtype0
?
Xmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean
VariableV2"/device:GPU:0*
shape:3*
shared_name *k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean*
dtype0*
	container 
?
_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/AssignAssignXmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_meanjmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean*
validate_shape(
?
]main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/readIdentityXmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean"/device:GPU:0*
T0*k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean
?
}main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/shapeConst*
dtype0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
valueB"   3   
?
{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/minConst*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
valueB
 *???*
dtype0
?
{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/maxConst*
dtype0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
valueB
 *??<
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/RandomUniformRandomUniform}main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/shape*
T0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
dtype0*
seed2 *

seed 
?
{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/subSub{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/max{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/min*
T0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev
?
{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/mulMul?main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/RandomUniform{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/sub*
T0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev
?
wmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniformAdd{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/mul{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/min*
T0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev
?
\main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev
VariableV2"/device:GPU:0*
shared_name *o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
dtype0*
	container *
shape:	?3
?
cmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/AssignAssign\main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddevwmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
validate_shape(
?
amain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/readIdentity\main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev"/device:GPU:0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
T0
?
{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/shapeConst*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev*
valueB:3*
dtype0
?
ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/minConst*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev*
valueB
 *???*
dtype0
?
ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/maxConst*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev*
valueB
 *??<*
dtype0
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/RandomUniformRandomUniform{main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/shape*

seed *
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev*
dtype0*
seed2 
?
ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/subSubymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/maxymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/min*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev
?
ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/mulMul?main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/RandomUniformymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/sub*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev
?
umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniformAddymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/mulymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/min*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev
?
Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev
VariableV2"/device:GPU:0*
shared_name *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev*
dtype0*
	container *
shape:3
?
amain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/AssignAssignZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddevumain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform"/device:GPU:0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev*
validate_shape(*
use_locking(*
T0
?
_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/readIdentityZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev"/device:GPU:0*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev
?
bmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal/shapeConst"/device:GPU:0*
valueB:3*
dtype0
?
amain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
cmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
qmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal/RandomStandardNormalRandomStandardNormalbmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal/shape"/device:GPU:0*
dtype0*
seed2 *

seed *
T0
?
`main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal/mulMulqmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal/RandomStandardNormalcmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal/stddev"/device:GPU:0*
T0
?
\main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normalAdd`main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal/mulamain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal/mean"/device:GPU:0*
T0
?
Rmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/AbsAbs\main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal"/device:GPU:0*
T0
?
Smain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/SqrtSqrtRmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/Abs"/device:GPU:0*
T0
?
Smain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/SignSign\main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal"/device:GPU:0*
T0
?
Rmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/mulMulSmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/SqrtSmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/Sign"/device:GPU:0*
T0
?
dmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1/shapeConst"/device:GPU:0*
valueB"      *
dtype0
?
cmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
emain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
smain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1/RandomStandardNormalRandomStandardNormaldmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1/shape"/device:GPU:0*
T0*
dtype0*
seed2 *

seed 
?
bmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1/mulMulsmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1/RandomStandardNormalemain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1/stddev"/device:GPU:0*
T0
?
^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1Addbmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1/mulcmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1/mean"/device:GPU:0*
T0
?
Tmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/Abs_1Abs^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1"/device:GPU:0*
T0
?
Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/Sqrt_1SqrtTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/Abs_1"/device:GPU:0*
T0
?
Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/Sign_1Sign^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1"/device:GPU:0*
T0
?
Tmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/mul_1MulUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/Sqrt_1Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/Sign_1"/device:GPU:0*
T0
?
dmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2/shapeConst"/device:GPU:0*
valueB"   3   *
dtype0
?
cmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
emain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
smain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2/RandomStandardNormalRandomStandardNormaldmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2/shape"/device:GPU:0*
dtype0*
seed2 *

seed *
T0
?
bmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2/mulMulsmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2/RandomStandardNormalemain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2/stddev"/device:GPU:0*
T0
?
^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2Addbmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2/mulcmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2/mean"/device:GPU:0*
T0
?
Tmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/Abs_2Abs^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2"/device:GPU:0*
T0
?
Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/Sqrt_2SqrtTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/Abs_2"/device:GPU:0*
T0
?
Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/Sign_2Sign^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2"/device:GPU:0*
T0
?
Tmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/mul_2MulUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/Sqrt_2Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/Sign_2"/device:GPU:0*
T0
?
Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/MatMulMatMulTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/mul_1Tmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/mul_2"/device:GPU:0*
transpose_a( *
transpose_b( *
T0
?
Pmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_2Mul_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/readRmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/mul"/device:GPU:0*
T0
?
Pmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_3Add]main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/readPmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_2"/device:GPU:0*
T0
?
Pmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_3Mulamain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/readUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/MatMul"/device:GPU:0*
T0
?
Pmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_4Add_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/readPmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_3"/device:GPU:0*
T0
?
Smain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_1MatMulOmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ReluPmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_4"/device:GPU:0*
transpose_a( *
transpose_b( *
T0
?
Pmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5AddSmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_1Pmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_3"/device:GPU:0*
T0
?
Ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDims/dimConst"/device:GPU:0*
value	B :*
dtype0
?
Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDims
ExpandDimsPmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5Ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDims/dim"/device:GPU:0*

Tdim0*
T0
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/shapeConst*
dtype0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
valueB"      
?
~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/minConst*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
valueB
 *  ??*
dtype0
?
~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/maxConst*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
valueB
 *  ?=*
dtype0
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/RandomUniformRandomUniform?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/shape*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
dtype0*
seed2 *

seed 
?
~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/subSub~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/max~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/min*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean
?
~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/mulMul?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/RandomUniform~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/sub*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean
?
zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniformAdd~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/mul~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/min*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean
?
_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean
VariableV2"/device:GPU:0*
dtype0*
	container *
shape:
??*
shared_name *r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean
?
fmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/AssignAssign_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_meanzmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
validate_shape(
?
dmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/readIdentity_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean"/device:GPU:0*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean
?
omain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Initializer/zerosConst*
dtype0*p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean*
valueB?*    
?
]main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean
VariableV2"/device:GPU:0*p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean*
dtype0*
	container *
shape:?*
shared_name 
?
dmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/AssignAssign]main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_meanomain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Initializer/zeros"/device:GPU:0*
T0*p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean*
validate_shape(*
use_locking(
?
bmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/readIdentity]main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean"/device:GPU:0*
T0*p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/shapeConst*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
valueB"      *
dtype0
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/minConst*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
valueB
 *   ?*
dtype0
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/maxConst*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
valueB
 *   =*
dtype0
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/RandomUniformRandomUniform?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/shape*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
dtype0*
seed2 *

seed *
T0
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/subSub?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/max?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/min*
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/mulMul?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/RandomUniform?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/sub*
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev
?
|main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniformAdd?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/mul?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/min*
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev
?
amain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev
VariableV2"/device:GPU:0*
shape:
??*
shared_name *t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
dtype0*
	container 
?
hmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/AssignAssignamain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev|main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform"/device:GPU:0*
validate_shape(*
use_locking(*
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev
?
fmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/readIdentityamain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev"/device:GPU:0*
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/shapeConst*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
valueB:?*
dtype0
?
~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/minConst*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
valueB
 *   ?*
dtype0
?
~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/maxConst*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
valueB
 *   =*
dtype0
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/RandomUniformRandomUniform?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/shape*

seed *
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
dtype0*
seed2 
?
~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/subSub~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/max~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/min*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev
?
~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/mulMul?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/RandomUniform~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/sub*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev
?
zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniformAdd~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/mul~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/min*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev
?
_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev
VariableV2"/device:GPU:0*
	container *
shape:?*
shared_name *r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
dtype0
?
fmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/AssignAssign_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddevzmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
validate_shape(
?
dmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/readIdentity_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev"/device:GPU:0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
T0
?
gmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal/shapeConst"/device:GPU:0*
dtype0*
valueB:?
?
fmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
hmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
vmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal/RandomStandardNormalRandomStandardNormalgmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal/shape"/device:GPU:0*
T0*
dtype0*
seed2 *

seed 
?
emain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal/mulMulvmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal/RandomStandardNormalhmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal/stddev"/device:GPU:0*
T0
?
amain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normalAddemain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal/mulfmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal/mean"/device:GPU:0*
T0
?
Wmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/AbsAbsamain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal"/device:GPU:0*
T0
?
Xmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/SqrtSqrtWmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/Abs"/device:GPU:0*
T0
?
Xmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/SignSignamain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal"/device:GPU:0*
T0
?
Wmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/mulMulXmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/SqrtXmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/Sign"/device:GPU:0*
T0
?
imain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1/shapeConst"/device:GPU:0*
valueB"      *
dtype0
?
hmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
jmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
xmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1/RandomStandardNormalRandomStandardNormalimain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1/shape"/device:GPU:0*
T0*
dtype0*
seed2 *

seed 
?
gmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1/mulMulxmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1/RandomStandardNormaljmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1/stddev"/device:GPU:0*
T0
?
cmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1Addgmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1/mulhmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1/mean"/device:GPU:0*
T0
?
Ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/Abs_1Abscmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1"/device:GPU:0*
T0
?
Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/Sqrt_1SqrtYmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/Abs_1"/device:GPU:0*
T0
?
Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/Sign_1Signcmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1"/device:GPU:0*
T0
?
Ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/mul_1MulZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/Sqrt_1Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/Sign_1"/device:GPU:0*
T0
?
imain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2/shapeConst"/device:GPU:0*
valueB"      *
dtype0
?
hmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
jmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
xmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2/RandomStandardNormalRandomStandardNormalimain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2/shape"/device:GPU:0*
T0*
dtype0*
seed2 *

seed 
?
gmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2/mulMulxmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2/RandomStandardNormaljmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2/stddev"/device:GPU:0*
T0
?
cmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2Addgmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2/mulhmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2/mean"/device:GPU:0*
T0
?
Ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/Abs_2Abscmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2"/device:GPU:0*
T0
?
Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/Sqrt_2SqrtYmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/Abs_2"/device:GPU:0*
T0
?
Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/Sign_2Signcmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2"/device:GPU:0*
T0
?
Ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/mul_2MulZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/Sqrt_2Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/Sign_2"/device:GPU:0*
T0
?
Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/MatMulMatMulYmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/mul_1Ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/mul_2"/device:GPU:0*
transpose_a( *
transpose_b( *
T0
?
Smain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mulMuldmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/readWmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/mul"/device:GPU:0*
T0
?
Smain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/addAddbmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/readSmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul"/device:GPU:0*
T0
?
Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_1Mulfmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/readZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/MatMul"/device:GPU:0*
T0
?
Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_1Adddmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/readUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_1"/device:GPU:0*
T0
?
Vmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMulMatMulLmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_sliceUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_1"/device:GPU:0*
transpose_a( *
transpose_b( *
T0
?
Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2AddVmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMulSmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add"/device:GPU:0*
T0
?
Tmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/ReluReluUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2"/device:GPU:0*
T0
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/shapeConst*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
valueB"   ?   *
dtype0
?
~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/minConst*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
valueB
 *?5?*
dtype0
?
~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/maxConst*
dtype0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
valueB
 *?5=
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/RandomUniformRandomUniform?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/shape*
seed2 *

seed *
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
dtype0
?
~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/subSub~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/max~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/min*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean
?
~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/mulMul?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/RandomUniform~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/sub*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean
?
zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniformAdd~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/mul~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/min*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean
?
_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean
VariableV2"/device:GPU:0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
dtype0*
	container *
shape:
??*
shared_name 
?
fmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/AssignAssign_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_meanzmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
validate_shape(
?
dmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/readIdentity_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean"/device:GPU:0*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean
?
omain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Initializer/zerosConst*p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean*
valueB?*    *
dtype0
?
]main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean
VariableV2"/device:GPU:0*
shape:?*
shared_name *p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean*
dtype0*
	container 
?
dmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/AssignAssign]main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_meanomain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean*
validate_shape(
?
bmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/readIdentity]main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean"/device:GPU:0*
T0*p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/shapeConst*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
valueB"   ?   *
dtype0
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/minConst*
dtype0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
valueB
 *???
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/maxConst*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
valueB
 *??<*
dtype0
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/RandomUniformRandomUniform?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/shape*
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
dtype0*
seed2 *

seed 
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/subSub?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/max?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/min*
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/mulMul?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/RandomUniform?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/sub*
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev
?
|main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniformAdd?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/mul?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/min*
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev
?
amain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev
VariableV2"/device:GPU:0*
shared_name *t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
dtype0*
	container *
shape:
??
?
hmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/AssignAssignamain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev|main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform"/device:GPU:0*
validate_shape(*
use_locking(*
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev
?
fmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/readIdentityamain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev"/device:GPU:0*
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/shapeConst*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
valueB:?*
dtype0
?
~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/minConst*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
valueB
 *???*
dtype0
?
~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/maxConst*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
valueB
 *??<*
dtype0
?
?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/RandomUniformRandomUniform?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/shape*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
dtype0*
seed2 *

seed *
T0
?
~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/subSub~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/max~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/min*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev
?
~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/mulMul?main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/RandomUniform~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/sub*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
T0
?
zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniformAdd~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/mul~main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/min*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev
?
_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev
VariableV2"/device:GPU:0*
shape:?*
shared_name *r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
dtype0*
	container 
?
fmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/AssignAssign_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddevzmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform"/device:GPU:0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
validate_shape(*
use_locking(*
T0
?
dmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/readIdentity_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev"/device:GPU:0*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev
?
gmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal/shapeConst"/device:GPU:0*
valueB:?*
dtype0
?
fmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
hmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
vmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal/RandomStandardNormalRandomStandardNormalgmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal/shape"/device:GPU:0*

seed *
T0*
dtype0*
seed2 
?
emain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal/mulMulvmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal/RandomStandardNormalhmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal/stddev"/device:GPU:0*
T0
?
amain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normalAddemain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal/mulfmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal/mean"/device:GPU:0*
T0
?
Wmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/AbsAbsamain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal"/device:GPU:0*
T0
?
Xmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/SqrtSqrtWmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/Abs"/device:GPU:0*
T0
?
Xmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/SignSignamain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal"/device:GPU:0*
T0
?
Wmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/mulMulXmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/SqrtXmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/Sign"/device:GPU:0*
T0
?
imain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1/shapeConst"/device:GPU:0*
valueB"      *
dtype0
?
hmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
jmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
xmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1/RandomStandardNormalRandomStandardNormalimain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1/shape"/device:GPU:0*
T0*
dtype0*
seed2 *

seed 
?
gmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1/mulMulxmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1/RandomStandardNormaljmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1/stddev"/device:GPU:0*
T0
?
cmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1Addgmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1/mulhmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1/mean"/device:GPU:0*
T0
?
Ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/Abs_1Abscmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1"/device:GPU:0*
T0
?
Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/Sqrt_1SqrtYmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/Abs_1"/device:GPU:0*
T0
?
Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/Sign_1Signcmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1"/device:GPU:0*
T0
?
Ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/mul_1MulZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/Sqrt_1Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/Sign_1"/device:GPU:0*
T0
?
imain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2/shapeConst"/device:GPU:0*
dtype0*
valueB"   ?   
?
hmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
jmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
xmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2/RandomStandardNormalRandomStandardNormalimain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2/shape"/device:GPU:0*
dtype0*
seed2 *

seed *
T0
?
gmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2/mulMulxmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2/RandomStandardNormaljmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2/stddev"/device:GPU:0*
T0
?
cmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2Addgmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2/mulhmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2/mean"/device:GPU:0*
T0
?
Ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/Abs_2Abscmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2"/device:GPU:0*
T0
?
Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/Sqrt_2SqrtYmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/Abs_2"/device:GPU:0*
T0
?
Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/Sign_2Signcmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2"/device:GPU:0*
T0
?
Ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/mul_2MulZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/Sqrt_2Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/Sign_2"/device:GPU:0*
T0
?
Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/MatMulMatMulYmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/mul_1Ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/mul_2"/device:GPU:0*
T0*
transpose_a( *
transpose_b( 
?
Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_2Muldmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/readWmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/mul"/device:GPU:0*
T0
?
Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_3Addbmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/readUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_2"/device:GPU:0*
T0
?
Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_3Mulfmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/readZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/MatMul"/device:GPU:0*
T0
?
Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_4Adddmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/readUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_3"/device:GPU:0*
T0
?
Xmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1MatMulTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/ReluUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_4"/device:GPU:0*
transpose_a( *
transpose_b( *
T0
?
Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5AddXmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_3"/device:GPU:0*
T0
?
Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/ShapeShapeLmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice"/device:GPU:0*
out_type0*
T0
?
cmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/strided_slice/stackConst"/device:GPU:0*
valueB: *
dtype0
?
emain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/strided_slice/stack_1Const"/device:GPU:0*
valueB:*
dtype0
?
emain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/strided_slice/stack_2Const"/device:GPU:0*
valueB:*
dtype0
?
]main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/strided_sliceStridedSliceUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Shapecmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/strided_slice/stackemain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/strided_slice/stack_1emain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/strided_slice/stack_2"/device:GPU:0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
?
_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape/shape/1Const"/device:GPU:0*
value	B :*
dtype0
?
_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape/shape/2Const"/device:GPU:0*
value	B :3*
dtype0
?
]main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape/shapePack]main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/strided_slice_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape/shape/1_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape/shape/2"/device:GPU:0*
T0*

axis *
N
?
Wmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/ReshapeReshapeUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5]main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape/shape"/device:GPU:0*
T0*
Tshape0
?
fmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean/reduction_indicesConst"/device:GPU:0*
value	B :*
dtype0
?
Tmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MeanMeanWmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshapefmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean/reduction_indices"/device:GPU:0*

Tidx0*
	keep_dims(*
T0
?
Smain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/subSubWmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/ReshapeTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean"/device:GPU:0*
T0
?
Emain_level/agent/main/online/network_0/rainbow_q_values_head_0/outputAddUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDimsSmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub"/device:GPU:0*
T0
?
Fmain_level/agent/main/online/network_0/rainbow_q_values_head_0/SoftmaxSoftmaxEmain_level/agent/main/online/network_0/rainbow_q_values_head_0/output"/device:GPU:0*
T0
?
Lmain_level/agent/main/online/network_0/rainbow_q_values_head_0/distributionsPlaceholder"/device:GPU:0*
dtype0* 
shape:?????????3
?
xmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradientLmain_level/agent/main/online/network_0/rainbow_q_values_head_0/distributions"/device:GPU:0*
T0
?
hmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/RankConst"/device:GPU:0*
value	B :*
dtype0
?
imain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/ShapeShapeEmain_level/agent/main/online/network_0/rainbow_q_values_head_0/output"/device:GPU:0*
T0*
out_type0
?
jmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Rank_1Const"/device:GPU:0*
value	B :*
dtype0
?
kmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Shape_1ShapeEmain_level/agent/main/online/network_0/rainbow_q_values_head_0/output"/device:GPU:0*
T0*
out_type0
?
imain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub/yConst"/device:GPU:0*
value	B :*
dtype0
?
gmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/SubSubjmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Rank_1imain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub/y"/device:GPU:0*
T0
?
omain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice/beginPackgmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub"/device:GPU:0*
N*
T0*

axis 
?
nmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice/sizeConst"/device:GPU:0*
valueB:*
dtype0
?
imain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/SliceSlicekmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Shape_1omain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice/beginnmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice/size"/device:GPU:0*
T0*
Index0
?
smain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat/values_0Const"/device:GPU:0*
valueB:
?????????*
dtype0
?
omain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat/axisConst"/device:GPU:0*
value	B : *
dtype0
?
jmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concatConcatV2smain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat/values_0imain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sliceomain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat/axis"/device:GPU:0*
T0*
N*

Tidx0
?
kmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/ReshapeReshapeEmain_level/agent/main/online/network_0/rainbow_q_values_head_0/outputjmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat"/device:GPU:0*
T0*
Tshape0
?
jmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Rank_2Const"/device:GPU:0*
value	B :*
dtype0
?
kmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Shape_2Shapexmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/labels_stop_gradient"/device:GPU:0*
T0*
out_type0
?
kmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub_1/yConst"/device:GPU:0*
value	B :*
dtype0
?
imain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub_1Subjmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Rank_2kmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub_1/y"/device:GPU:0*
T0
?
qmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_1/beginPackimain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub_1"/device:GPU:0*
T0*

axis *
N
?
pmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst"/device:GPU:0*
valueB:*
dtype0
?
kmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_1Slicekmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Shape_2qmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_1/beginpmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_1/size"/device:GPU:0*
T0*
Index0
?
umain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat_1/values_0Const"/device:GPU:0*
dtype0*
valueB:
?????????
?
qmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat_1/axisConst"/device:GPU:0*
value	B : *
dtype0
?
lmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat_1ConcatV2umain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat_1/values_0kmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_1qmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat_1/axis"/device:GPU:0*
T0*
N*

Tidx0
?
mmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_1Reshapexmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/labels_stop_gradientlmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat_1"/device:GPU:0*
T0*
Tshape0
?
cmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogitskmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshapemmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_1"/device:GPU:0*
T0
?
kmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub_2/yConst"/device:GPU:0*
value	B :*
dtype0
?
imain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub_2Subhmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Rankkmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub_2/y"/device:GPU:0*
T0
?
qmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_2/beginConst"/device:GPU:0*
valueB: *
dtype0
?
pmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_2/sizePackimain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub_2"/device:GPU:0*
T0*

axis *
N
?
kmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_2Sliceimain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Shapeqmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_2/beginpmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_2/size"/device:GPU:0*
T0*
Index0
?
mmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_2Reshapecmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sgkmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_2"/device:GPU:0*
T0*
Tshape0
?
Cmain_level/agent/main/online/network_0/rainbow_q_values_head_0/CastCastFmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax"/device:GPU:0*

SrcT0*
Truncate( *

DstT0
?
Mmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/axesConst"/device:GPU:0*
valueB:*
dtype0
?
Mmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/freeConst"/device:GPU:0*
valueB"       *
dtype0
?
Nmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/ShapeShapeCmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Cast"/device:GPU:0*
T0*
out_type0
?
Vmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/GatherV2/axisConst"/device:GPU:0*
dtype0*
value	B : 
?
Qmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/GatherV2GatherV2Nmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/ShapeMmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/freeVmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/GatherV2/axis"/device:GPU:0*
Taxis0*
Tindices0*
Tparams0
?
Xmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/GatherV2_1/axisConst"/device:GPU:0*
dtype0*
value	B : 
?
Smain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/GatherV2_1GatherV2Nmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/ShapeMmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/axesXmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/GatherV2_1/axis"/device:GPU:0*
Tindices0*
Tparams0*
Taxis0
?
Nmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/ConstConst"/device:GPU:0*
valueB: *
dtype0
?
Mmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/ProdProdQmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/GatherV2Nmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/Const"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
Pmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/Const_1Const"/device:GPU:0*
valueB: *
dtype0
?
Omain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/Prod_1ProdSmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/GatherV2_1Pmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/Const_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
Tmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/concat/axisConst"/device:GPU:0*
value	B : *
dtype0
?
Omain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/concatConcatV2Mmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/freeMmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/axesTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/concat/axis"/device:GPU:0*
T0*
N*

Tidx0
?
Nmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/stackPackMmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/ProdOmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/Prod_1"/device:GPU:0*
T0*

axis *
N
?
Rmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/transpose	TransposeCmain_level/agent/main/online/network_0/rainbow_q_values_head_0/CastOmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/concat"/device:GPU:0*
T0*
Tperm0
?
Pmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/ReshapeReshapeRmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/transposeNmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/stack"/device:GPU:0*
T0*
Tshape0
?
Ymain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/transpose_1/permConst"/device:GPU:0*
dtype0*
valueB: 
?
Tmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/transpose_1	Transpose+main_level/agent/main/online/network_0/CastYmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/transpose_1/perm"/device:GPU:0*
Tperm0*
T0
?
Xmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/Reshape_1/shapeConst"/device:GPU:0*
valueB"3      *
dtype0
?
Rmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/Reshape_1ReshapeTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/transpose_1Xmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/Reshape_1/shape"/device:GPU:0*
T0*
Tshape0
?
Omain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/MatMulMatMulPmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/ReshapeRmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/Reshape_1"/device:GPU:0*
T0*
transpose_a( *
transpose_b( 
?
Pmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/Const_2Const"/device:GPU:0*
valueB *
dtype0
?
Vmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/concat_1/axisConst"/device:GPU:0*
value	B : *
dtype0
?
Qmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/concat_1ConcatV2Qmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/GatherV2Pmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/Const_2Vmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/concat_1/axis"/device:GPU:0*
T0*
N*

Tidx0
?
Hmain_level/agent/main/online/network_0/rainbow_q_values_head_0/TensordotReshapeOmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/MatMulQmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Tensordot/concat_1"/device:GPU:0*
T0*
Tshape0
?
Hmain_level/agent/main/online/network_0/rainbow_q_values_head_0/truediv/yConst"/device:GPU:0*
valueB 2      ??*
dtype0
?
Fmain_level/agent/main/online/network_0/rainbow_q_values_head_0/truedivRealDivHmain_level/agent/main/online/network_0/rainbow_q_values_head_0/TensordotHmain_level/agent/main/online/network_0/rainbow_q_values_head_0/truediv/y"/device:GPU:0*
T0
?
Hmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_1SoftmaxFmain_level/agent/main/online/network_0/rainbow_q_values_head_0/truediv"/device:GPU:0*
T0
?
hmain_level/agent/main/online/network_0/rainbow_q_values_head_0/rainbow_q_values_head_0_importance_weightPlaceholder"/device:GPU:0* 
shape:?????????*
dtype0
?
(main_level/agent/main/online/Rank/packedPackmmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_2"/device:GPU:0*

axis *
N*
T0
Z
!main_level/agent/main/online/RankConst"/device:GPU:0*
value	B :*
dtype0
a
(main_level/agent/main/online/range/startConst"/device:GPU:0*
value	B : *
dtype0
a
(main_level/agent/main/online/range/deltaConst"/device:GPU:0*
value	B :*
dtype0
?
"main_level/agent/main/online/rangeRange(main_level/agent/main/online/range/start!main_level/agent/main/online/Rank(main_level/agent/main/online/range/delta"/device:GPU:0*

Tidx0
?
&main_level/agent/main/online/Sum/inputPackmmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_2"/device:GPU:0*
T0*

axis *
N
?
 main_level/agent/main/online/SumSum&main_level/agent/main/online/Sum/input"main_level/agent/main/online/range"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
g
%main_level/agent/main/online/0_holderPlaceholder"/device:GPU:0*
dtype0*
shape:
??
b
%main_level/agent/main/online/1_holderPlaceholder"/device:GPU:0*
dtype0*
shape:?
g
%main_level/agent/main/online/2_holderPlaceholder"/device:GPU:0*
dtype0*
shape:
??
b
%main_level/agent/main/online/3_holderPlaceholder"/device:GPU:0*
dtype0*
shape:?
]
%main_level/agent/main/online/4_holderPlaceholder"/device:GPU:0*
shape: *
dtype0
g
%main_level/agent/main/online/5_holderPlaceholder"/device:GPU:0*
dtype0*
shape:
??
b
%main_level/agent/main/online/6_holderPlaceholder"/device:GPU:0*
shape:?*
dtype0
g
%main_level/agent/main/online/7_holderPlaceholder"/device:GPU:0*
dtype0*
shape:
??
b
%main_level/agent/main/online/8_holderPlaceholder"/device:GPU:0*
dtype0*
shape:?
f
%main_level/agent/main/online/9_holderPlaceholder"/device:GPU:0*
dtype0*
shape:	?3
b
&main_level/agent/main/online/10_holderPlaceholder"/device:GPU:0*
dtype0*
shape:3
g
&main_level/agent/main/online/11_holderPlaceholder"/device:GPU:0*
shape:	?3*
dtype0
b
&main_level/agent/main/online/12_holderPlaceholder"/device:GPU:0*
dtype0*
shape:3
h
&main_level/agent/main/online/13_holderPlaceholder"/device:GPU:0*
dtype0*
shape:
??
c
&main_level/agent/main/online/14_holderPlaceholder"/device:GPU:0*
dtype0*
shape:?
h
&main_level/agent/main/online/15_holderPlaceholder"/device:GPU:0*
dtype0*
shape:
??
c
&main_level/agent/main/online/16_holderPlaceholder"/device:GPU:0*
shape:?*
dtype0
h
&main_level/agent/main/online/17_holderPlaceholder"/device:GPU:0*
shape:
??*
dtype0
c
&main_level/agent/main/online/18_holderPlaceholder"/device:GPU:0*
dtype0*
shape:?
h
&main_level/agent/main/online/19_holderPlaceholder"/device:GPU:0*
shape:
??*
dtype0
c
&main_level/agent/main/online/20_holderPlaceholder"/device:GPU:0*
dtype0*
shape:?
?
%main_level/agent/main/online/Assign_1AssignNmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean%main_level/agent/main/online/0_holder"/device:GPU:0*
use_locking( *
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean*
validate_shape(
?
%main_level/agent/main/online/Assign_2AssignLmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean%main_level/agent/main/online/1_holder"/device:GPU:0*
use_locking( *
T0*_
_classU
SQloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean*
validate_shape(
?
%main_level/agent/main/online/Assign_3AssignPmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev%main_level/agent/main/online/2_holder"/device:GPU:0*
use_locking( *
T0*c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev*
validate_shape(
?
%main_level/agent/main/online/Assign_4AssignNmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev%main_level/agent/main/online/3_holder"/device:GPU:0*
validate_shape(*
use_locking( *
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev
?
%main_level/agent/main/online/Assign_5AssignHmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers%main_level/agent/main/online/4_holder"/device:GPU:0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
validate_shape(*
use_locking( *
T0
?
%main_level/agent/main/online/Assign_6AssignZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean%main_level/agent/main/online/5_holder"/device:GPU:0*
use_locking( *
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
validate_shape(
?
%main_level/agent/main/online/Assign_7AssignXmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean%main_level/agent/main/online/6_holder"/device:GPU:0*
use_locking( *
T0*k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean*
validate_shape(
?
%main_level/agent/main/online/Assign_8Assign\main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev%main_level/agent/main/online/7_holder"/device:GPU:0*
use_locking( *
T0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
validate_shape(
?
%main_level/agent/main/online/Assign_9AssignZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev%main_level/agent/main/online/8_holder"/device:GPU:0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev*
validate_shape(*
use_locking( *
T0
?
&main_level/agent/main/online/Assign_10AssignZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean%main_level/agent/main/online/9_holder"/device:GPU:0*
validate_shape(*
use_locking( *
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean
?
&main_level/agent/main/online/Assign_11AssignXmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean&main_level/agent/main/online/10_holder"/device:GPU:0*
use_locking( *
T0*k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean*
validate_shape(
?
&main_level/agent/main/online/Assign_12Assign\main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev&main_level/agent/main/online/11_holder"/device:GPU:0*
use_locking( *
T0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
validate_shape(
?
&main_level/agent/main/online/Assign_13AssignZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev&main_level/agent/main/online/12_holder"/device:GPU:0*
use_locking( *
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev*
validate_shape(
?
&main_level/agent/main/online/Assign_14Assign_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean&main_level/agent/main/online/13_holder"/device:GPU:0*
use_locking( *
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
validate_shape(
?
&main_level/agent/main/online/Assign_15Assign]main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean&main_level/agent/main/online/14_holder"/device:GPU:0*
use_locking( *
T0*p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean*
validate_shape(
?
&main_level/agent/main/online/Assign_16Assignamain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev&main_level/agent/main/online/15_holder"/device:GPU:0*
use_locking( *
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
validate_shape(
?
&main_level/agent/main/online/Assign_17Assign_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev&main_level/agent/main/online/16_holder"/device:GPU:0*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
validate_shape(*
use_locking( 
?
&main_level/agent/main/online/Assign_18Assign_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean&main_level/agent/main/online/17_holder"/device:GPU:0*
use_locking( *
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
validate_shape(
?
&main_level/agent/main/online/Assign_19Assign]main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean&main_level/agent/main/online/18_holder"/device:GPU:0*
validate_shape(*
use_locking( *
T0*p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean
?
&main_level/agent/main/online/Assign_20Assignamain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev&main_level/agent/main/online/19_holder"/device:GPU:0*
use_locking( *
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
validate_shape(
?
&main_level/agent/main/online/Assign_21Assign_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev&main_level/agent/main/online/20_holder"/device:GPU:0*
use_locking( *
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
validate_shape(
d
,main_level/agent/main/online/gradients/ShapeConst"/device:GPU:0*
valueB *
dtype0
l
0main_level/agent/main/online/gradients/grad_ys_0Const"/device:GPU:0*
valueB
 *  ??*
dtype0
?
+main_level/agent/main/online/gradients/FillFill,main_level/agent/main/online/gradients/Shape0main_level/agent/main/online/gradients/grad_ys_0"/device:GPU:0*

index_type0*
T0
?
Zmain_level/agent/main/online/gradients/main_level/agent/main/online/Sum_grad/Reshape/shapeConst"/device:GPU:0*!
valueB"         *
dtype0
?
Tmain_level/agent/main/online/gradients/main_level/agent/main/online/Sum_grad/ReshapeReshape+main_level/agent/main/online/gradients/FillZmain_level/agent/main/online/gradients/main_level/agent/main/online/Sum_grad/Reshape/shape"/device:GPU:0*
T0*
Tshape0
?
Rmain_level/agent/main/online/gradients/main_level/agent/main/online/Sum_grad/ShapeShape&main_level/agent/main/online/Sum/input"/device:GPU:0*
T0*
out_type0
?
Qmain_level/agent/main/online/gradients/main_level/agent/main/online/Sum_grad/TileTileTmain_level/agent/main/online/gradients/main_level/agent/main/online/Sum_grad/ReshapeRmain_level/agent/main/online/gradients/main_level/agent/main/online/Sum_grad/Shape"/device:GPU:0*

Tmultiples0*
T0
?
Zmain_level/agent/main/online/gradients/main_level/agent/main/online/Sum/input_grad/unstackUnpackQmain_level/agent/main/online/gradients/main_level/agent/main/online/Sum_grad/Tile"/device:GPU:0*
T0*	
num*

axis 
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShapecmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg"/device:GPU:0*
out_type0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshapeZmain_level/agent/main/online/gradients/main_level/agent/main/online/Sum/input_grad/unstack?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
1main_level/agent/main/online/gradients/zeros_like	ZerosLikeemain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg:1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst"/device:GPU:0*
valueB :
?????????*
dtype0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDims?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Reshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim"/device:GPU:0*
T0*

Tdim0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/mulMul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/ExpandDimsemain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg:1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmaxkmain_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/NegNeg?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst"/device:GPU:0*
valueB :
?????????*
dtype0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDims?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Reshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim"/device:GPU:0*

Tdim0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/mul_1Mul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/Neg"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapeEmain_level/agent/main/online/network_0/rainbow_q_values_head_0/output"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/mul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
wmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/ShapeShapeUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDims"/device:GPU:0*
out_type0*
T0
?
ymain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Shape_1ShapeSmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/BroadcastGradientArgsBroadcastGradientArgswmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Shapeymain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Shape_1"/device:GPU:0*
T0
?
umain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/SumSum?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
ymain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/ReshapeReshapeumain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Sumwmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Shape"/device:GPU:0*
Tshape0*
T0
?
wmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Sum_1Sum?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
{main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Reshape_1Reshapewmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Sum_1ymain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Shape_1"/device:GPU:0*
Tshape0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/ShapeShapePmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/ReshapeReshapeymain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Reshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/Shape"/device:GPU:0*
Tshape0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/ShapeShapeWmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape"/device:GPU:0*
out_type0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape_1ShapeTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/SumSum{main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Reshape_1?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/BroadcastGradientArgs"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/ReshapeReshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Sum?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Sum_1Sum{main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Reshape_1?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/NegNeg?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Sum_1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape_1Reshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Neg?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/ShapeShapeSmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_1"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape_1Const"/device:GPU:0*
valueB:3*
dtype0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/SumSum?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/Reshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/ReshapeReshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Sum?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Sum_1Sum?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/Reshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape_1Reshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Sum_1?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ShapeShapeWmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/SizeConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :*
dtype0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/addAddfmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean/reduction_indices?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Size"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/modFloorMod?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/add?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Size"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_1Const"/device:GPU:0*
dtype0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
valueB 
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/startConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B : *
dtype0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/deltaConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :*
dtype0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/rangeRange?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/start?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Size?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/delta"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*

Tidx0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Fill/valueConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :*
dtype0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/FillFill?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_1?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Fill/value"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*

index_type0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/DynamicStitchDynamicStitch?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/mod?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Fill"/device:GPU:0*
N*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum/yConst"/device:GPU:0*
dtype0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/MaximumMaximum?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/DynamicStitch?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum/y"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordivFloorDiv?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ReshapeReshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape_1?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/DynamicStitch"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/TileTile?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Reshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordiv"/device:GPU:0*

Tmultiples0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_2ShapeWmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_3ShapeTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ConstConst"/device:GPU:0*
valueB: *
dtype0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ProdProd?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_2?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Const_1Const"/device:GPU:0*
valueB: *
dtype0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Prod_1Prod?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_3?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Const_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1/yConst"/device:GPU:0*
value	B :*
dtype0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1Maximum?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Prod_1?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1/y"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordiv_1FloorDiv?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Prod?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/CastCast?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordiv_1"/device:GPU:0*

SrcT0*
Truncate( *

DstT0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/truedivRealDiv?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Tile?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Cast"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMulMatMul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/ReshapePmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_4"/device:GPU:0*
transpose_a( *
transpose_b(*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMul_1MatMulOmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/Relu?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape"/device:GPU:0*
transpose_a(*
transpose_b( *
T0
?
+main_level/agent/main/online/gradients/AddNAddN?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/truediv"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape*
N
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/ShapeShapeUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/ReshapeReshape+main_level/agent/main/online/gradients/AddN?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/Relu_grad/ReluGradReluGrad?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMulOmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/Relu"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_2_grad/MulMul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape_1Rmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/mul"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_2_grad/Mul_1Mul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape_1_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/ShapeShapeXmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape_1Const"/device:GPU:0*
valueB:?*
dtype0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/SumSum?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/Reshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/BroadcastGradientArgs"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/ReshapeReshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Sum?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Sum_1Sum?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/Reshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape_1Reshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Sum_1?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/ShapeShapeQmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape_1Const"/device:GPU:0*
valueB:?*
dtype0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/SumSum?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/Relu_grad/ReluGrad?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/BroadcastGradientArgs"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/ReshapeReshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Sum?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape"/device:GPU:0*
Tshape0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Sum_1Sum?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/Relu_grad/ReluGrad?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape_1Reshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Sum_1?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_3_grad/MulMul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMul_1Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/MatMul"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_3_grad/Mul_1Mul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMul_1amain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMulMatMul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/ReshapeUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_4"/device:GPU:0*
transpose_a( *
transpose_b(*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMul_1MatMulTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Relu?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape"/device:GPU:0*
transpose_a(*
transpose_b( *
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMulMatMul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/ReshapePmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_1"/device:GPU:0*
transpose_a( *
transpose_b(*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul_1MatMulLmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape"/device:GPU:0*
T0*
transpose_a(*
transpose_b( 
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Relu_grad/ReluGradReluGrad?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMulTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Relu"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_2_grad/MulMul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape_1Wmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/mul"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_2_grad/Mul_1Mul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape_1dmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/read"/device:GPU:0*
T0
?
~main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_grad/MulMul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape_1Rmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/mul"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_grad/Mul_1Mul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape_1_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/ShapeShapeVmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul"/device:GPU:0*
out_type0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape_1Const"/device:GPU:0*
dtype0*
valueB:?
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/SumSum?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Relu_grad/ReluGrad?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/BroadcastGradientArgs"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/ReshapeReshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Sum?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Sum_1Sum?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Relu_grad/ReluGrad?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape_1Reshape?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Sum_1?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_3_grad/MulMul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMul_1Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/MatMul"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_3_grad/Mul_1Mul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMul_1fmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_1_grad/MulMul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul_1Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/MatMul"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_1_grad/Mul_1Mul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul_1amain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMulMatMul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/ReshapeUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_1"/device:GPU:0*
transpose_a( *
transpose_b(*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul_1MatMulLmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape"/device:GPU:0*
transpose_a(*
transpose_b( *
T0
?
-main_level/agent/main/online/gradients/AddN_1AddN?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul*
N
?
~main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice_grad/ShapeShape*main_level/agent/main/online/network_0/add"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice_grad/StridedSliceGradStridedSliceGrad~main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice_grad/ShapeRmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice/stackTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice/stack_1Tmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice/stack_2-main_level/agent/main/online/gradients/AddN_1"/device:GPU:0*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_grad/MulMul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape_1Wmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/mul"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_grad/Mul_1Mul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape_1dmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/read"/device:GPU:0*
T0
?
\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/ShapeShape*main_level/agent/main/online/network_0/mul"/device:GPU:0*
T0*
out_type0
?
^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Shape_1Shape,main_level/agent/main/online/network_0/mul_1"/device:GPU:0*
T0*
out_type0
?
lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/BroadcastGradientArgsBroadcastGradientArgs\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Shape^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Shape_1"/device:GPU:0*
T0
?
Zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/SumSum?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice_grad/StridedSliceGradlmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/ReshapeReshapeZmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Sum\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Shape"/device:GPU:0*
Tshape0*
T0
?
\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Sum_1Sum?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice_grad/StridedSliceGradnmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
`main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Reshape_1Reshape\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Sum_1^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_1_grad/MulMul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul_1Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/MatMul"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_1_grad/Mul_1Mul?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul_1fmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/read"/device:GPU:0*
T0
?
\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
?
^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Shape_1Shape3main_level/agent/main/online/network_0/StopGradient"/device:GPU:0*
T0*
out_type0
?
lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/BroadcastGradientArgsBroadcastGradientArgs\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Shape^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Shape_1"/device:GPU:0*
T0
?
Zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/MulMul^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Reshape3main_level/agent/main/online/network_0/StopGradient"/device:GPU:0*
T0
?
Zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/SumSumZmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Mullmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/ReshapeReshapeZmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Sum\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Mul_1Mul*main_level/agent/main/online/network_0/sub^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Reshape"/device:GPU:0*
T0
?
\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Sum_1Sum\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Mul_1nmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
`main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Reshape_1Reshape\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Sum_1^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
?
`main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Shape_1Shape.main_level/agent/main/online/network_0/mul_1/y"/device:GPU:0*
T0*
out_type0
?
nmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Shape`main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0
?
\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/MulMul`main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Reshape_1.main_level/agent/main/online/network_0/mul_1/y"/device:GPU:0*
T0
?
\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/SumSum\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Mulnmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
`main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/ReshapeReshape\main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Sum^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Mul_1MulMmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/read`main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/add_grad/Reshape_1"/device:GPU:0*
T0
?
^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Sum_1Sum^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Mul_1pmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
bmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Reshape_1Reshape^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Sum_1`main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
Zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/sub_grad/NegNeg^main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_grad/Reshape"/device:GPU:0*
T0
?
bmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1/y_grad/unstackUnpackbmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Reshape_1"/device:GPU:0*
T0*	
num*

axis 
?
-main_level/agent/main/online/gradients/AddN_2AddN`main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/ReshapeZmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/sub_grad/Neg"/device:GPU:0*
T0*s
_classi
geloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Reshape*
N
?
|main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/ShapeShapeZmain_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0
?
~main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/ReshapeReshapebmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1/y_grad/unstack|main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation_grad/ReluGradReluGrad~main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/ReshapeZmain_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0
?
jmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/add_2_grad/ShapeShape9main_level/agent/main/online/network_0/observation/MatMul"/device:GPU:0*
T0*
out_type0
?
lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/add_2_grad/Shape_1Const"/device:GPU:0*
valueB:?*
dtype0
?
zmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsjmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/add_2_grad/Shapelmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/add_2_grad/Shape_1"/device:GPU:0*
T0
?
hmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/add_2_grad/SumSum?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation_grad/ReluGradzmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/add_2_grad/BroadcastGradientArgs"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/add_2_grad/ReshapeReshapehmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/add_2_grad/Sumjmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/add_2_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
jmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/add_2_grad/Sum_1Sum?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation_grad/ReluGrad|main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/add_2_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
nmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/add_2_grad/Reshape_1Reshapejmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/add_2_grad/Sum_1lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/add_2_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
lmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/MatMul_grad/MatMulMatMullmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/add_2_grad/Reshape8main_level/agent/main/online/network_0/observation/add_1"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
?
nmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/MatMul_grad/MatMul_1MatMul6main_level/agent/main/online/network_0/observation/sublmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/add_2_grad/Reshape"/device:GPU:0*
T0*
transpose_a(*
transpose_b( 
?
fmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/mul_grad/MulMulnmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/add_2_grad/Reshape_1Fmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/mul"/device:GPU:0*
T0
?
hmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/mul_grad/Mul_1Mulnmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/add_2_grad/Reshape_1Smain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/read"/device:GPU:0*
T0
?
hmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/mul_1_grad/MulMulnmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/MatMul_grad/MatMul_1Imain_level/agent/main/online/network_0/observation/NoisyNetDense_0/MatMul"/device:GPU:0*
T0
?
jmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/mul_1_grad/Mul_1Mulnmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/MatMul_grad/MatMul_1Umain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/read"/device:GPU:0*
T0
?
/main_level/agent/main/online/global_norm/L2LossL2Lossnmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/MatMul_grad/MatMul_1"/device:GPU:0*
T0*?
_classw
usloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/MatMul_grad/MatMul_1
?
1main_level/agent/main/online/global_norm/L2Loss_1L2Lossnmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/add_2_grad/Reshape_1"/device:GPU:0*
T0*?
_classw
usloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/add_2_grad/Reshape_1
?
1main_level/agent/main/online/global_norm/L2Loss_2L2Losshmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/mul_1_grad/Mul"/device:GPU:0*
T0*{
_classq
omloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/mul_1_grad/Mul
?
1main_level/agent/main/online/global_norm/L2Loss_3L2Lossfmain_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/mul_grad/Mul"/device:GPU:0*
T0*y
_classo
mkloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/observation/mul_grad/Mul
?
1main_level/agent/main/online/global_norm/L2Loss_4L2Loss-main_level/agent/main/online/gradients/AddN_2"/device:GPU:0*
T0*s
_classi
geloc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/mul_1_grad/Reshape
?
1main_level/agent/main/online/global_norm/L2Loss_5L2Loss?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul_1"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul_1
?
1main_level/agent/main/online/global_norm/L2Loss_6L2Loss?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape_1"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape_1
?
1main_level/agent/main/online/global_norm/L2Loss_7L2Loss?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_1_grad/Mul"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_1_grad/Mul
?
1main_level/agent/main/online/global_norm/L2Loss_8L2Loss~main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_grad/Mul"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_grad/Mul
?
1main_level/agent/main/online/global_norm/L2Loss_9L2Loss?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMul_1"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMul_1*
T0
?
2main_level/agent/main/online/global_norm/L2Loss_10L2Loss?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape_1"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape_1*
T0
?
2main_level/agent/main/online/global_norm/L2Loss_11L2Loss?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_3_grad/Mul"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_3_grad/Mul
?
2main_level/agent/main/online/global_norm/L2Loss_12L2Loss?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_2_grad/Mul"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_2_grad/Mul
?
2main_level/agent/main/online/global_norm/L2Loss_13L2Loss?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul_1"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul_1
?
2main_level/agent/main/online/global_norm/L2Loss_14L2Loss?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape_1"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape_1
?
2main_level/agent/main/online/global_norm/L2Loss_15L2Loss?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_1_grad/Mul"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_1_grad/Mul*
T0
?
2main_level/agent/main/online/global_norm/L2Loss_16L2Loss?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_grad/Mul"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_grad/Mul
?
2main_level/agent/main/online/global_norm/L2Loss_17L2Loss?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMul_1"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMul_1
?
2main_level/agent/main/online/global_norm/L2Loss_18L2Loss?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape_1"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape_1
?
2main_level/agent/main/online/global_norm/L2Loss_19L2Loss?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_3_grad/Mul"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_3_grad/Mul*
T0
?
2main_level/agent/main/online/global_norm/L2Loss_20L2Loss?main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_2_grad/Mul"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_2_grad/Mul
?	
.main_level/agent/main/online/global_norm/stackPack/main_level/agent/main/online/global_norm/L2Loss1main_level/agent/main/online/global_norm/L2Loss_11main_level/agent/main/online/global_norm/L2Loss_21main_level/agent/main/online/global_norm/L2Loss_31main_level/agent/main/online/global_norm/L2Loss_41main_level/agent/main/online/global_norm/L2Loss_51main_level/agent/main/online/global_norm/L2Loss_61main_level/agent/main/online/global_norm/L2Loss_71main_level/agent/main/online/global_norm/L2Loss_81main_level/agent/main/online/global_norm/L2Loss_92main_level/agent/main/online/global_norm/L2Loss_102main_level/agent/main/online/global_norm/L2Loss_112main_level/agent/main/online/global_norm/L2Loss_122main_level/agent/main/online/global_norm/L2Loss_132main_level/agent/main/online/global_norm/L2Loss_142main_level/agent/main/online/global_norm/L2Loss_152main_level/agent/main/online/global_norm/L2Loss_162main_level/agent/main/online/global_norm/L2Loss_172main_level/agent/main/online/global_norm/L2Loss_182main_level/agent/main/online/global_norm/L2Loss_192main_level/agent/main/online/global_norm/L2Loss_20"/device:GPU:0*

axis *
N*
T0
k
.main_level/agent/main/online/global_norm/ConstConst"/device:GPU:0*
valueB: *
dtype0
?
,main_level/agent/main/online/global_norm/SumSum.main_level/agent/main/online/global_norm/stack.main_level/agent/main/online/global_norm/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
l
0main_level/agent/main/online/global_norm/Const_1Const"/device:GPU:0*
valueB
 *   @*
dtype0
?
,main_level/agent/main/online/global_norm/mulMul,main_level/agent/main/online/global_norm/Sum0main_level/agent/main/online/global_norm/Const_1"/device:GPU:0*
T0
?
4main_level/agent/main/online/global_norm/global_normSqrt,main_level/agent/main/online/global_norm/mul"/device:GPU:0*
T0
?
.main_level/agent/main/online/gradients_1/ShapeShapeFmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax"/device:GPU:0*
T0*
out_type0
n
2main_level/agent/main/online/gradients_1/grad_ys_0Const"/device:GPU:0*
valueB
 *  ??*
dtype0
?
-main_level/agent/main/online/gradients_1/FillFill.main_level/agent/main/online/gradients_1/Shape2main_level/agent/main/online/gradients_1/grad_ys_0"/device:GPU:0*

index_type0*
T0
?
xmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/mulMul-main_level/agent/main/online/gradients_1/FillFmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/Sum/reduction_indicesConst"/device:GPU:0*
dtype0*
valueB :
?????????
?
xmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/SumSumxmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/mul?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/Sum/reduction_indices"/device:GPU:0*
T0*

Tidx0*
	keep_dims(
?
xmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/subSub-main_level/agent/main/online/gradients_1/Fillxmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/Sum"/device:GPU:0*
T0
?
zmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/mul_1Mulxmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/subFmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax"/device:GPU:0*
T0
?
ymain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/ShapeShapeUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDims"/device:GPU:0*
T0*
out_type0
?
{main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Shape_1ShapeSmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/BroadcastGradientArgsBroadcastGradientArgsymain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Shape{main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Shape_1"/device:GPU:0*
T0
?
wmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/SumSumzmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/mul_1?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/BroadcastGradientArgs"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
{main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/ReshapeReshapewmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Sumymain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Shape"/device:GPU:0*
Tshape0*
T0
?
ymain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Sum_1Sumzmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/mul_1?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
}main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Reshape_1Reshapeymain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Sum_1{main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/ShapeShapePmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/ReshapeReshape{main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Reshape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/ShapeShapeWmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape_1ShapeTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/SumSum}main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Reshape_1?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/ReshapeReshape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Sum?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Sum_1Sum}main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Reshape_1?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/NegNeg?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Sum_1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape_1Reshape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Neg?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/ShapeShapeSmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_1"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape_1Const"/device:GPU:0*
valueB:3*
dtype0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/SumSum?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/Reshape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/BroadcastGradientArgs"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/ReshapeReshape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Sum?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Sum_1Sum?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/Reshape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape_1Reshape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Sum_1?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ShapeShapeWmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/SizeConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :*
dtype0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/addAddfmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean/reduction_indices?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Size"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/modFloorMod?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/add?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Size"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_1Const"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
valueB *
dtype0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/startConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B : *
dtype0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/deltaConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :*
dtype0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/rangeRange?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/start?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Size?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/delta"/device:GPU:0*

Tidx0*?
_class?
??loc:@main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Fill/valueConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :*
dtype0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/FillFill?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_1?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Fill/value"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*

index_type0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/DynamicStitchDynamicStitch?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/mod?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Fill"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
N
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum/yConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :*
dtype0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/MaximumMaximum?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/DynamicStitch?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum/y"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordivFloorDiv?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ReshapeReshape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape_1?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/DynamicStitch"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/TileTile?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Reshape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordiv"/device:GPU:0*

Tmultiples0*
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_2ShapeWmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_3ShapeTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean"/device:GPU:0*
out_type0*
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ConstConst"/device:GPU:0*
valueB: *
dtype0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ProdProd?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_2?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Const_1Const"/device:GPU:0*
valueB: *
dtype0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Prod_1Prod?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_3?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Const_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1/yConst"/device:GPU:0*
value	B :*
dtype0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1Maximum?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Prod_1?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1/y"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordiv_1FloorDiv?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Prod?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/CastCast?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordiv_1"/device:GPU:0*

SrcT0*
Truncate( *

DstT0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/truedivRealDiv?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Tile?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Cast"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMulMatMul?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/ReshapePmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_4"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMul_1MatMulOmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/Relu?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
?
-main_level/agent/main/online/gradients_1/AddNAddN?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/truediv"/device:GPU:0*
N*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/ShapeShapeUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/ReshapeReshape-main_level/agent/main/online/gradients_1/AddN?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/Relu_grad/ReluGradReluGrad?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMulOmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/Relu"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/ShapeShapeXmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape_1Const"/device:GPU:0*
valueB:?*
dtype0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/SumSum?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/Reshape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/BroadcastGradientArgs"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/ReshapeReshape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Sum?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Sum_1Sum?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/Reshape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape_1Reshape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Sum_1?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/ShapeShapeQmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape_1Const"/device:GPU:0*
valueB:?*
dtype0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/SumSum?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/Relu_grad/ReluGrad?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/BroadcastGradientArgs"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/ReshapeReshape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Sum?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Sum_1Sum?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/Relu_grad/ReluGrad?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape_1Reshape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Sum_1?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMulMatMul?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/ReshapeUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_4"/device:GPU:0*
T0*
transpose_a( *
transpose_b(
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMul_1MatMulTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Relu?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape"/device:GPU:0*
T0*
transpose_a(*
transpose_b( 
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMulMatMul?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/ReshapePmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_1"/device:GPU:0*
transpose_a( *
transpose_b(*
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul_1MatMulLmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape"/device:GPU:0*
T0*
transpose_a(*
transpose_b( 
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Relu_grad/ReluGradReluGrad?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMulTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Relu"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/ShapeShapeVmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape_1Const"/device:GPU:0*
valueB:?*
dtype0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/SumSum?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Relu_grad/ReluGrad?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/ReshapeReshape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Sum?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Sum_1Sum?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Relu_grad/ReluGrad?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape_1Reshape?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Sum_1?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMulMatMul?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/ReshapeUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_1"/device:GPU:0*
transpose_a( *
transpose_b(*
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul_1MatMulLmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape"/device:GPU:0*
transpose_a(*
transpose_b( *
T0
?
/main_level/agent/main/online/gradients_1/AddN_1AddN?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul*
N
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice_grad/ShapeShape*main_level/agent/main/online/network_0/add"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice_grad/StridedSliceGradStridedSliceGrad?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice_grad/ShapeRmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice/stackTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice/stack_1Tmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice/stack_2/main_level/agent/main/online/gradients_1/AddN_1"/device:GPU:0*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask
?
^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/ShapeShape*main_level/agent/main/online/network_0/mul"/device:GPU:0*
T0*
out_type0
?
`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Shape_1Shape,main_level/agent/main/online/network_0/mul_1"/device:GPU:0*
T0*
out_type0
?
nmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Shape`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Shape_1"/device:GPU:0*
T0
?
\main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/SumSum?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice_grad/StridedSliceGradnmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/ReshapeReshape\main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Sum^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Sum_1Sum?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice_grad/StridedSliceGradpmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
bmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Reshape_1Reshape^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Sum_1`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
?
`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Shape_1Shape3main_level/agent/main/online/network_0/StopGradient"/device:GPU:0*
T0*
out_type0
?
nmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Shape`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Shape_1"/device:GPU:0*
T0
?
\main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/MulMul`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Reshape3main_level/agent/main/online/network_0/StopGradient"/device:GPU:0*
T0
?
\main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/SumSum\main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Mulnmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/ReshapeReshape\main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Sum^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Mul_1Mul*main_level/agent/main/online/network_0/sub`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Reshape"/device:GPU:0*
T0
?
^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Sum_1Sum^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Mul_1pmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
bmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Reshape_1Reshape^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Sum_1`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/ShapeConst"/device:GPU:0*
dtype0*
valueB 
?
bmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Shape_1Shape.main_level/agent/main/online/network_0/mul_1/y"/device:GPU:0*
T0*
out_type0
?
pmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Shapebmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0
?
^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/MulMulbmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Reshape_1.main_level/agent/main/online/network_0/mul_1/y"/device:GPU:0*
T0
?
^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/SumSum^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Mulpmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
bmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/ReshapeReshape^main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Sum`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Shape"/device:GPU:0*
Tshape0*
T0
?
`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Mul_1MulMmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/readbmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/add_grad/Reshape_1"/device:GPU:0*
T0
?
`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Sum_1Sum`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Mul_1rmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
dmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Reshape_1Reshape`main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Sum_1bmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
dmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1/y_grad/unstackUnpackdmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1_grad/Reshape_1"/device:GPU:0*

axis *
T0*	
num
?
~main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/ShapeShapeZmain_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
out_type0*
T0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/ReshapeReshapedmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/mul_1/y_grad/unstack~main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation_grad/ReluGradReluGrad?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/ReshapeZmain_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0
?
lmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/add_2_grad/ShapeShape9main_level/agent/main/online/network_0/observation/MatMul"/device:GPU:0*
T0*
out_type0
?
nmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/add_2_grad/Shape_1Const"/device:GPU:0*
dtype0*
valueB:?
?
|main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/add_2_grad/BroadcastGradientArgsBroadcastGradientArgslmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/add_2_grad/Shapenmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/add_2_grad/Shape_1"/device:GPU:0*
T0
?
jmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/add_2_grad/SumSum?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation_grad/ReluGrad|main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/add_2_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
nmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/add_2_grad/ReshapeReshapejmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/add_2_grad/Sumlmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/add_2_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
lmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/add_2_grad/Sum_1Sum?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation_grad/ReluGrad~main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/add_2_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
pmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/add_2_grad/Reshape_1Reshapelmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/add_2_grad/Sum_1nmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/add_2_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
nmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/MatMul_grad/MatMulMatMulnmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/add_2_grad/Reshape8main_level/agent/main/online/network_0/observation/add_1"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
?
pmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/MatMul_grad/MatMul_1MatMul6main_level/agent/main/online/network_0/observation/subnmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/add_2_grad/Reshape"/device:GPU:0*
transpose_a(*
transpose_b( *
T0
?
jmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/ShapeShape:main_level/agent/main/online/network_0/observation/truediv"/device:GPU:0*
T0*
out_type0
?
lmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Shape_1Const"/device:GPU:0*
valueB *
dtype0
?
zmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/BroadcastGradientArgsBroadcastGradientArgsjmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Shapelmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Shape_1"/device:GPU:0*
T0
?
hmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/SumSumnmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/MatMul_grad/MatMulzmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/BroadcastGradientArgs"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
lmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/ReshapeReshapehmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Sumjmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
jmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Sum_1Sumnmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/MatMul_grad/MatMul|main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
hmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/NegNegjmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Sum_1"/device:GPU:0*
T0
?
nmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Reshape_1Reshapehmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Neglmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
nmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/ShapeShape>main_level/agent/main/online/network_0/observation/observation"/device:GPU:0*
T0*
out_type0
?
pmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/Shape_1Const"/device:GPU:0*
valueB *
dtype0
?
~main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsnmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/Shapepmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/Shape_1"/device:GPU:0*
T0
?
pmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/RealDivRealDivlmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Reshape<main_level/agent/main/online/network_0/observation/truediv/y"/device:GPU:0*
T0
?
lmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/SumSumpmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/RealDiv~main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
pmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/ReshapeReshapelmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/Sumnmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
lmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/NegNeg>main_level/agent/main/online/network_0/observation/observation"/device:GPU:0*
T0
?
rmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/RealDiv_1RealDivlmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/Neg<main_level/agent/main/online/network_0/observation/truediv/y"/device:GPU:0*
T0
?
rmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/RealDiv_2RealDivrmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/RealDiv_1<main_level/agent/main/online/network_0/observation/truediv/y"/device:GPU:0*
T0
?
lmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/mulMullmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/sub_grad/Reshapermain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/RealDiv_2"/device:GPU:0*
T0
?
nmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/Sum_1Sumlmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/mul?main_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
rmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/Reshape_1Reshapenmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/Sum_1pmain_level/agent/main/online/gradients_1/main_level/agent/main/online/network_0/observation/truediv_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
4main_level/agent/main/online/output_gradient_weightsPlaceholder"/device:GPU:0*
dtype0* 
shape:?????????3
?
2main_level/agent/main/online/gradients_2/grad_ys_0Identity4main_level/agent/main/online/output_gradient_weights"/device:GPU:0*
T0
?
xmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/mulMul2main_level/agent/main/online/gradients_2/grad_ys_0Fmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/Sum/reduction_indicesConst"/device:GPU:0*
valueB :
?????????*
dtype0
?
xmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/SumSumxmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/mul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/Sum/reduction_indices"/device:GPU:0*

Tidx0*
	keep_dims(*
T0
?
xmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/subSub2main_level/agent/main/online/gradients_2/grad_ys_0xmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/Sum"/device:GPU:0*
T0
?
zmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/mul_1Mulxmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/subFmain_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax"/device:GPU:0*
T0
?
ymain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/ShapeShapeUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDims"/device:GPU:0*
T0*
out_type0
?
{main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Shape_1ShapeSmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/BroadcastGradientArgsBroadcastGradientArgsymain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Shape{main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Shape_1"/device:GPU:0*
T0
?
wmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/SumSumzmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/mul_1?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/BroadcastGradientArgs"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
{main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/ReshapeReshapewmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Sumymain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
ymain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Sum_1Sumzmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/Softmax_grad/mul_1?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
}main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Reshape_1Reshapeymain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Sum_1{main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Shape_1"/device:GPU:0*
Tshape0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/ShapeShapePmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/ReshapeReshape{main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Reshape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/Shape"/device:GPU:0*
Tshape0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/ShapeShapeWmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape_1ShapeTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean"/device:GPU:0*
out_type0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/SumSum}main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Reshape_1?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/ReshapeReshape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Sum?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Sum_1Sum}main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/output_grad/Reshape_1?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/NegNeg?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Sum_1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape_1Reshape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Neg?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/ShapeShapeSmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_1"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape_1Const"/device:GPU:0*
valueB:3*
dtype0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/SumSum?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/Reshape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/ReshapeReshape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Sum?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Sum_1Sum?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/Reshape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape_1Reshape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Sum_1?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ShapeShapeWmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/SizeConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :*
dtype0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/addAddfmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean/reduction_indices?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Size"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/modFloorMod?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/add?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Size"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_1Const"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
valueB *
dtype0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/startConst"/device:GPU:0*
dtype0*?
_class?
??loc:@main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B : 
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/deltaConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :*
dtype0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/rangeRange?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/start?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Size?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/delta"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*

Tidx0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Fill/valueConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :*
dtype0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/FillFill?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_1?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Fill/value"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*

index_type0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/DynamicStitchDynamicStitch?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/mod?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Fill"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
N
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum/yConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :*
dtype0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/MaximumMaximum?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/DynamicStitch?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum/y"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordivFloorDiv?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ReshapeReshape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape_1?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/DynamicStitch"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/TileTile?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Reshape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordiv"/device:GPU:0*

Tmultiples0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_2ShapeWmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_3ShapeTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ConstConst"/device:GPU:0*
valueB: *
dtype0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ProdProd?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_2?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Const"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Const_1Const"/device:GPU:0*
dtype0*
valueB: 
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Prod_1Prod?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_3?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Const_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1/yConst"/device:GPU:0*
value	B :*
dtype0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1Maximum?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Prod_1?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1/y"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordiv_1FloorDiv?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Prod?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/CastCast?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordiv_1"/device:GPU:0*

SrcT0*
Truncate( *

DstT0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/truedivRealDiv?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Tile?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Cast"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMulMatMul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/ReshapePmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_4"/device:GPU:0*
transpose_a( *
transpose_b(*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMul_1MatMulOmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/Relu?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape"/device:GPU:0*
T0*
transpose_a(*
transpose_b( 
?
-main_level/agent/main/online/gradients_2/AddNAddN?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/truediv"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape*
N
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/ShapeShapeUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/ReshapeReshape-main_level/agent/main/online/gradients_2/AddN?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/Relu_grad/ReluGradReluGrad?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMulOmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/Relu"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_2_grad/MulMul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape_1Rmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/mul"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_2_grad/Mul_1Mul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape_1_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/ShapeShapeXmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape_1Const"/device:GPU:0*
valueB:?*
dtype0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/SumSum?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/Reshape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/BroadcastGradientArgs"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/ReshapeReshape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Sum?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Sum_1Sum?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/Reshape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape_1Reshape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Sum_1?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/ShapeShapeQmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape_1Const"/device:GPU:0*
valueB:?*
dtype0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/SumSum?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/Relu_grad/ReluGrad?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/ReshapeReshape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Sum?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Sum_1Sum?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/Relu_grad/ReluGrad?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape_1Reshape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Sum_1?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_3_grad/MulMul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMul_1Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/MatMul"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_3_grad/Mul_1Mul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMul_1amain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMulMatMul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/ReshapeUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_4"/device:GPU:0*
T0*
transpose_a( *
transpose_b(
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMul_1MatMulTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Relu?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape"/device:GPU:0*
T0*
transpose_a(*
transpose_b( 
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMulMatMul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/ReshapePmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_1"/device:GPU:0*
transpose_a( *
transpose_b(*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul_1MatMulLmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape"/device:GPU:0*
T0*
transpose_a(*
transpose_b( 
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Relu_grad/ReluGradReluGrad?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMulTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Relu"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_2_grad/MulMul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape_1Wmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/mul"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_2_grad/Mul_1Mul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape_1dmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_grad/MulMul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape_1Rmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/mul"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_grad/Mul_1Mul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape_1_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/ShapeShapeVmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape_1Const"/device:GPU:0*
dtype0*
valueB:?
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/SumSum?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Relu_grad/ReluGrad?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/ReshapeReshape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Sum?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Sum_1Sum?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/Relu_grad/ReluGrad?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape_1Reshape?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Sum_1?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_3_grad/MulMul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMul_1Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/MatMul"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_3_grad/Mul_1Mul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMul_1fmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_1_grad/MulMul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul_1Umain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/MatMul"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/mul_1_grad/Mul_1Mul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul_1amain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMulMatMul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/ReshapeUmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_1"/device:GPU:0*
transpose_a( *
transpose_b(*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul_1MatMulLmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape"/device:GPU:0*
T0*
transpose_a(*
transpose_b( 
?
/main_level/agent/main/online/gradients_2/AddN_1AddN?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul*
N
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice_grad/ShapeShape*main_level/agent/main/online/network_0/add"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice_grad/StridedSliceGradStridedSliceGrad?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice_grad/ShapeRmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice/stackTmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice/stack_1Tmain_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice/stack_2/main_level/agent/main/online/gradients_2/AddN_1"/device:GPU:0*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_grad/MulMul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape_1Wmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/mul"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_grad/Mul_1Mul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape_1dmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/read"/device:GPU:0*
T0
?
^main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/add_grad/ShapeShape*main_level/agent/main/online/network_0/mul"/device:GPU:0*
T0*
out_type0
?
`main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/add_grad/Shape_1Shape,main_level/agent/main/online/network_0/mul_1"/device:GPU:0*
T0*
out_type0
?
nmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/add_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/add_grad/Shape`main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/add_grad/Shape_1"/device:GPU:0*
T0
?
\main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/add_grad/SumSum?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice_grad/StridedSliceGradnmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
`main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/add_grad/ReshapeReshape\main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/add_grad/Sum^main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/add_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
^main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/add_grad/Sum_1Sum?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/strided_slice_grad/StridedSliceGradpmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/add_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
bmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/add_grad/Reshape_1Reshape^main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/add_grad/Sum_1`main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_1_grad/MulMul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul_1Zmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/MatMul"/device:GPU:0*
T0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/mul_1_grad/Mul_1Mul?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul_1fmain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/read"/device:GPU:0*
T0
?
^main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
?
`main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_grad/Shape_1Shape3main_level/agent/main/online/network_0/StopGradient"/device:GPU:0*
T0*
out_type0
?
nmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_grad/Shape`main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_grad/Shape_1"/device:GPU:0*
T0
?
\main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_grad/MulMul`main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/add_grad/Reshape3main_level/agent/main/online/network_0/StopGradient"/device:GPU:0*
T0
?
\main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_grad/SumSum\main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_grad/Mulnmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_grad/BroadcastGradientArgs"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
`main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_grad/ReshapeReshape\main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_grad/Sum^main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
^main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_grad/Mul_1Mul*main_level/agent/main/online/network_0/sub`main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/add_grad/Reshape"/device:GPU:0*
T0
?
^main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_grad/Sum_1Sum^main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_grad/Mul_1pmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
bmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_grad/Reshape_1Reshape^main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_grad/Sum_1`main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
`main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
?
bmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/Shape_1Shape.main_level/agent/main/online/network_0/mul_1/y"/device:GPU:0*
T0*
out_type0
?
pmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs`main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/Shapebmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0
?
^main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/MulMulbmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/add_grad/Reshape_1.main_level/agent/main/online/network_0/mul_1/y"/device:GPU:0*
T0
?
^main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/SumSum^main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/Mulpmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
bmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/ReshapeReshape^main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/Sum`main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
`main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/Mul_1MulMmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/readbmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/add_grad/Reshape_1"/device:GPU:0*
T0
?
`main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/Sum_1Sum`main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/Mul_1rmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
dmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/Reshape_1Reshape`main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/Sum_1bmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
\main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/sub_grad/NegNeg`main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_grad/Reshape"/device:GPU:0*
T0
?
dmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1/y_grad/unstackUnpackdmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/Reshape_1"/device:GPU:0*
T0*	
num*

axis 
?
/main_level/agent/main/online/gradients_2/AddN_2AddNbmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/Reshape\main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/sub_grad/Neg"/device:GPU:0*
T0*u
_classk
igloc:@main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1_grad/Reshape*
N
?
~main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/ShapeShapeZmain_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/ReshapeReshapedmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/mul_1/y_grad/unstack~main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation_grad/ReluGradReluGrad?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/Flatten/flatten/Reshape_grad/ReshapeZmain_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0
?
lmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/add_2_grad/ShapeShape9main_level/agent/main/online/network_0/observation/MatMul"/device:GPU:0*
T0*
out_type0
?
nmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/add_2_grad/Shape_1Const"/device:GPU:0*
valueB:?*
dtype0
?
|main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/add_2_grad/BroadcastGradientArgsBroadcastGradientArgslmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/add_2_grad/Shapenmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/add_2_grad/Shape_1"/device:GPU:0*
T0
?
jmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/add_2_grad/SumSum?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation_grad/ReluGrad|main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/add_2_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
nmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/add_2_grad/ReshapeReshapejmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/add_2_grad/Sumlmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/add_2_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
lmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/add_2_grad/Sum_1Sum?main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/BatchnormActivationDropout_1_activation_grad/ReluGrad~main_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/add_2_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
pmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/add_2_grad/Reshape_1Reshapelmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/add_2_grad/Sum_1nmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/add_2_grad/Shape_1"/device:GPU:0*
Tshape0*
T0
?
nmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/MatMul_grad/MatMulMatMulnmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/add_2_grad/Reshape8main_level/agent/main/online/network_0/observation/add_1"/device:GPU:0*
T0*
transpose_a( *
transpose_b(
?
pmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/MatMul_grad/MatMul_1MatMul6main_level/agent/main/online/network_0/observation/subnmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/add_2_grad/Reshape"/device:GPU:0*
T0*
transpose_a(*
transpose_b( 
?
hmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/mul_grad/MulMulpmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/add_2_grad/Reshape_1Fmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/mul"/device:GPU:0*
T0
?
jmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/mul_grad/Mul_1Mulpmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/add_2_grad/Reshape_1Smain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/read"/device:GPU:0*
T0
?
jmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/mul_1_grad/MulMulpmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/MatMul_grad/MatMul_1Imain_level/agent/main/online/network_0/observation/NoisyNetDense_0/MatMul"/device:GPU:0*
T0
?
lmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/mul_1_grad/Mul_1Mulpmain_level/agent/main/online/gradients_2/main_level/agent/main/online/network_0/observation/MatMul_grad/MatMul_1Umain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/read"/device:GPU:0*
T0
?
6main_level/agent/main/online/beta1_power/initial_valueConst"/device:GPU:0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
valueB
 *fff?*
dtype0
?
(main_level/agent/main/online/beta1_power
VariableV2"/device:GPU:0*
shape: *
shared_name *[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
dtype0*
	container 
?
/main_level/agent/main/online/beta1_power/AssignAssign(main_level/agent/main/online/beta1_power6main_level/agent/main/online/beta1_power/initial_value"/device:GPU:0*
validate_shape(*
use_locking(*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers
?
-main_level/agent/main/online/beta1_power/readIdentity(main_level/agent/main/online/beta1_power"/device:GPU:0*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers
?
6main_level/agent/main/online/beta2_power/initial_valueConst"/device:GPU:0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
valueB
 *?p}?*
dtype0
?
(main_level/agent/main/online/beta2_power
VariableV2"/device:GPU:0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
dtype0*
	container *
shape: *
shared_name 
?
/main_level/agent/main/online/beta2_power/AssignAssign(main_level/agent/main/online/beta2_power6main_level/agent/main/online/beta2_power/initial_value"/device:GPU:0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
validate_shape(*
use_locking(*
T0
?
-main_level/agent/main/online/beta2_power/readIdentity(main_level/agent/main/online/beta2_power"/device:GPU:0*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"H     *a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam/Initializer/zerosFill?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam/Initializer/zeros/shape_as_tensor?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean
?
pmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam
VariableV2"/device:GPU:0*
	container *
shape:
??*
shared_name *a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean*
dtype0
?
wmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam/AssignAssignpmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam/Initializer/zeros"/device:GPU:0*
validate_shape(*
use_locking(*
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean
?
umain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam/readIdentitypmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam"/device:GPU:0*
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"H     *a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam_1/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam_1/Initializer/zerosFill?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam_1/Initializer/zeros/shape_as_tensor?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam_1/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean
?
rmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam_1
VariableV2"/device:GPU:0*
shared_name *a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean*
dtype0*
	container *
shape:
??
?
ymain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam_1/AssignAssignrmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam_1?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean*
validate_shape(
?
wmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam_1/readIdentityrmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam_1"/device:GPU:0*
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Adam/Initializer/zerosConst"/device:GPU:0*
valueB?*    *_
_classU
SQloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean*
dtype0
?
nmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Adam
VariableV2"/device:GPU:0*
shape:?*
shared_name *_
_classU
SQloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean*
dtype0*
	container 
?
umain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Adam/AssignAssignnmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean*
validate_shape(
?
smain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Adam/readIdentitynmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Adam"/device:GPU:0*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB?*    *_
_classU
SQloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean*
dtype0
?
pmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Adam_1
VariableV2"/device:GPU:0*
shared_name *_
_classU
SQloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean*
dtype0*
	container *
shape:?
?
wmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Adam_1/AssignAssignpmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Adam_1?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean*
validate_shape(
?
umain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Adam_1/readIdentitypmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Adam_1"/device:GPU:0*
T0*_
_classU
SQloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"H     *c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam/Initializer/zerosFill?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam/Initializer/zeros/shape_as_tensor?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev
?
rmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam
VariableV2"/device:GPU:0*c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev*
dtype0*
	container *
shape:
??*
shared_name 
?
ymain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam/AssignAssignrmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev*
validate_shape(
?
wmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam/readIdentityrmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam"/device:GPU:0*
T0*c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"H     *c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam_1/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam_1/Initializer/zerosFill?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam_1/Initializer/zeros/shape_as_tensor?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam_1/Initializer/zeros/Const"/device:GPU:0*

index_type0*c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev*
T0
?
tmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam_1
VariableV2"/device:GPU:0*
	container *
shape:
??*
shared_name *c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev*
dtype0
?
{main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam_1/AssignAssigntmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam_1?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev*
validate_shape(
?
ymain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam_1/readIdentitytmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam_1"/device:GPU:0*
T0*c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Adam/Initializer/zerosConst"/device:GPU:0*
valueB?*    *a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev*
dtype0
?
pmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Adam
VariableV2"/device:GPU:0*
shared_name *a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev*
dtype0*
	container *
shape:?
?
wmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Adam/AssignAssignpmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev*
validate_shape(
?
umain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Adam/readIdentitypmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Adam"/device:GPU:0*
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB?*    *a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev*
dtype0
?
rmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Adam_1
VariableV2"/device:GPU:0*
shape:?*
shared_name *a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev*
dtype0*
	container 
?
ymain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Adam_1/AssignAssignrmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Adam_1?main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Adam_1/Initializer/zeros"/device:GPU:0*
validate_shape(*
use_locking(*
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev
?
wmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Adam_1/readIdentityrmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Adam_1"/device:GPU:0*
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev
?
|main_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam/Initializer/zerosConst"/device:GPU:0*
valueB
 *    *[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
dtype0
?
jmain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam
VariableV2"/device:GPU:0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
dtype0*
	container *
shape: *
shared_name 
?
qmain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam/AssignAssignjmain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam|main_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
validate_shape(
?
omain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam/readIdentityjmain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam"/device:GPU:0*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers
?
~main_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB
 *    *[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
dtype0
?
lmain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam_1
VariableV2"/device:GPU:0*
shared_name *[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
dtype0*
	container *
shape: 
?
smain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam_1/AssignAssignlmain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam_1~main_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
validate_shape(
?
qmain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam_1/readIdentitylmain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam_1"/device:GPU:0*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"      *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam/Initializer/zerosFill?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam/Initializer/zeros/shape_as_tensor?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean
?
|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam
VariableV2"/device:GPU:0*
shared_name *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
dtype0*
	container *
shape:
??
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam/AssignAssign|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam/readIdentity|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam"/device:GPU:0*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"      *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam_1/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam_1/Initializer/zerosFill?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam_1/Initializer/zeros/shape_as_tensor?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam_1/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean
?
~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam_1
VariableV2"/device:GPU:0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
dtype0*
	container *
shape:
??*
shared_name 
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam_1/AssignAssign~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam_1?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam_1/readIdentity~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam_1"/device:GPU:0*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Adam/Initializer/zerosConst"/device:GPU:0*
valueB?*    *k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean*
dtype0
?
zmain_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Adam
VariableV2"/device:GPU:0*
	container *
shape:?*
shared_name *k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Adam/AssignAssignzmain_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean*
validate_shape(
?
main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Adam/readIdentityzmain_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Adam"/device:GPU:0*
T0*k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB?*    *k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean*
dtype0
?
|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Adam_1
VariableV2"/device:GPU:0*
shape:?*
shared_name *k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean*
dtype0*
	container 
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Adam_1/AssignAssign|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Adam_1?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Adam_1/readIdentity|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Adam_1"/device:GPU:0*
T0*k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"      *o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam/Initializer/zerosFill?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam/Initializer/zeros/shape_as_tensor?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev
?
~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam
VariableV2"/device:GPU:0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
dtype0*
	container *
shape:
??*
shared_name 
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam/AssignAssign~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam/readIdentity~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam"/device:GPU:0*
T0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"      *o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam_1/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam_1/Initializer/zerosFill?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam_1/Initializer/zeros/shape_as_tensor?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam_1/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam_1
VariableV2"/device:GPU:0*
	container *
shape:
??*
shared_name *o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam_1/AssignAssign?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam_1?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam_1/Initializer/zeros"/device:GPU:0*
validate_shape(*
use_locking(*
T0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam_1/readIdentity?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam_1"/device:GPU:0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
T0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Adam/Initializer/zerosConst"/device:GPU:0*
valueB?*    *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev*
dtype0
?
|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Adam
VariableV2"/device:GPU:0*
shared_name *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev*
dtype0*
	container *
shape:?
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Adam/AssignAssign|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Adam/Initializer/zeros"/device:GPU:0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev*
validate_shape(*
use_locking(*
T0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Adam/readIdentity|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Adam"/device:GPU:0*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Adam_1/Initializer/zerosConst"/device:GPU:0*
dtype0*
valueB?*    *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev
?
~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Adam_1
VariableV2"/device:GPU:0*
	container *
shape:?*
shared_name *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Adam_1/AssignAssign~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Adam_1?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Adam_1/readIdentity~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Adam_1"/device:GPU:0*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"   3   *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam/Initializer/zeros/ConstConst"/device:GPU:0*
dtype0*
valueB
 *    *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam/Initializer/zerosFill?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam/Initializer/zeros/shape_as_tensor?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean
?
|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam
VariableV2"/device:GPU:0*
	container *
shape:	?3*
shared_name *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam/AssignAssign|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam/readIdentity|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam"/device:GPU:0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean*
T0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"   3   *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam_1/Initializer/zeros/ConstConst"/device:GPU:0*
dtype0*
valueB
 *    *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam_1/Initializer/zerosFill?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam_1/Initializer/zeros/shape_as_tensor?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam_1/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean
?
~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam_1
VariableV2"/device:GPU:0*
shared_name *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean*
dtype0*
	container *
shape:	?3
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam_1/AssignAssign~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam_1?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam_1/readIdentity~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam_1"/device:GPU:0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean*
T0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Adam/Initializer/zerosConst"/device:GPU:0*
valueB3*    *k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean*
dtype0
?
zmain_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Adam
VariableV2"/device:GPU:0*
shape:3*
shared_name *k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean*
dtype0*
	container 
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Adam/AssignAssignzmain_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean*
validate_shape(
?
main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Adam/readIdentityzmain_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Adam"/device:GPU:0*
T0*k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB3*    *k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean*
dtype0
?
|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Adam_1
VariableV2"/device:GPU:0*
	container *
shape:3*
shared_name *k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Adam_1/AssignAssign|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Adam_1?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Adam_1/Initializer/zeros"/device:GPU:0*k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean*
validate_shape(*
use_locking(*
T0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Adam_1/readIdentity|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Adam_1"/device:GPU:0*
T0*k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"   3   *o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam/Initializer/zerosFill?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam/Initializer/zeros/shape_as_tensor?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev
?
~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam
VariableV2"/device:GPU:0*
	container *
shape:	?3*
shared_name *o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam/AssignAssign~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam/readIdentity~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam"/device:GPU:0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
T0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"   3   *o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam_1/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam_1/Initializer/zerosFill?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam_1/Initializer/zeros/shape_as_tensor?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam_1/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam_1
VariableV2"/device:GPU:0*
dtype0*
	container *
shape:	?3*
shared_name *o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam_1/AssignAssign?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam_1?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam_1/readIdentity?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam_1"/device:GPU:0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
T0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Adam/Initializer/zerosConst"/device:GPU:0*
valueB3*    *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev*
dtype0
?
|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Adam
VariableV2"/device:GPU:0*
shared_name *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev*
dtype0*
	container *
shape:3
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Adam/AssignAssign|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Adam/readIdentity|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Adam"/device:GPU:0*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB3*    *m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev*
dtype0
?
~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Adam_1
VariableV2"/device:GPU:0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev*
dtype0*
	container *
shape:3*
shared_name 
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Adam_1/AssignAssign~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Adam_1?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Adam_1/Initializer/zeros"/device:GPU:0*
validate_shape(*
use_locking(*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Adam_1/readIdentity~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Adam_1"/device:GPU:0*
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"      *r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam/Initializer/zerosFill?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam/Initializer/zeros/shape_as_tensor?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam
VariableV2"/device:GPU:0*
	container *
shape:
??*
shared_name *r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam/AssignAssign?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam/Initializer/zeros"/device:GPU:0*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
validate_shape(*
use_locking(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam/readIdentity?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam"/device:GPU:0*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"      *r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam_1/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam_1/Initializer/zerosFill?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam_1/Initializer/zeros/shape_as_tensor?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam_1/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam_1
VariableV2"/device:GPU:0*
shape:
??*
shared_name *r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
dtype0*
	container 
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam_1/AssignAssign?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam_1?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam_1/readIdentity?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam_1"/device:GPU:0*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Adam/Initializer/zerosConst"/device:GPU:0*
valueB?*    *p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean*
dtype0
?
main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Adam
VariableV2"/device:GPU:0*
dtype0*
	container *
shape:?*
shared_name *p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Adam/AssignAssignmain_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Adam/readIdentitymain_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Adam"/device:GPU:0*
T0*p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB?*    *p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Adam_1
VariableV2"/device:GPU:0*
shared_name *p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean*
dtype0*
	container *
shape:?
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Adam_1/AssignAssign?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Adam_1?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Adam_1/readIdentity?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Adam_1"/device:GPU:0*
T0*p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"      *t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam/Initializer/zerosFill?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam/Initializer/zeros/shape_as_tensor?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam
VariableV2"/device:GPU:0*
shape:
??*
shared_name *t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
dtype0*
	container 
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam/AssignAssign?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam/Initializer/zeros"/device:GPU:0*
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
validate_shape(*
use_locking(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam/readIdentity?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam"/device:GPU:0*
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"      *t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam_1/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam_1/Initializer/zerosFill?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam_1/Initializer/zeros/shape_as_tensor?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam_1/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam_1
VariableV2"/device:GPU:0*
shared_name *t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
dtype0*
	container *
shape:
??
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam_1/AssignAssign?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam_1?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam_1/readIdentity?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam_1"/device:GPU:0*
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Adam/Initializer/zerosConst"/device:GPU:0*
valueB?*    *r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Adam
VariableV2"/device:GPU:0*
dtype0*
	container *
shape:?*
shared_name *r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Adam/AssignAssign?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Adam/readIdentity?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Adam"/device:GPU:0*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB?*    *r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Adam_1
VariableV2"/device:GPU:0*
	container *
shape:?*
shared_name *r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Adam_1/AssignAssign?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Adam_1?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Adam_1/readIdentity?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Adam_1"/device:GPU:0*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"   ?   *r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam/Initializer/zerosFill?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam/Initializer/zeros/shape_as_tensor?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam
VariableV2"/device:GPU:0*
shape:
??*
shared_name *r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
dtype0*
	container 
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam/AssignAssign?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam/readIdentity?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam"/device:GPU:0*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
dtype0*
valueB"   ?   *r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam_1/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam_1/Initializer/zerosFill?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam_1/Initializer/zeros/shape_as_tensor?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam_1/Initializer/zeros/Const"/device:GPU:0*

index_type0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
T0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam_1
VariableV2"/device:GPU:0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
dtype0*
	container *
shape:
??*
shared_name 
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam_1/AssignAssign?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam_1?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam_1/readIdentity?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam_1"/device:GPU:0*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Adam/Initializer/zerosConst"/device:GPU:0*
dtype0*
valueB?*    *p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean
?
main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Adam
VariableV2"/device:GPU:0*
shared_name *p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean*
dtype0*
	container *
shape:?
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Adam/AssignAssignmain_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Adam/Initializer/zeros"/device:GPU:0*
validate_shape(*
use_locking(*
T0*p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Adam/readIdentitymain_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Adam"/device:GPU:0*
T0*p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB?*    *p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Adam_1
VariableV2"/device:GPU:0*
dtype0*
	container *
shape:?*
shared_name *p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Adam_1/AssignAssign?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Adam_1?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Adam_1/readIdentity?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Adam_1"/device:GPU:0*
T0*p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"   ?   *t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam/Initializer/zerosFill?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam/Initializer/zeros/shape_as_tensor?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam
VariableV2"/device:GPU:0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
dtype0*
	container *
shape:
??*
shared_name 
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam/AssignAssign?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam/readIdentity?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam"/device:GPU:0*
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:0*
valueB"   ?   *t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam_1/Initializer/zeros/ConstConst"/device:GPU:0*
valueB
 *    *t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam_1/Initializer/zerosFill?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam_1/Initializer/zeros/shape_as_tensor?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam_1/Initializer/zeros/Const"/device:GPU:0*
T0*

index_type0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam_1
VariableV2"/device:GPU:0*
shape:
??*
shared_name *t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
dtype0*
	container 
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam_1/AssignAssign?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam_1?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam_1/readIdentity?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam_1"/device:GPU:0*
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Adam/Initializer/zerosConst"/device:GPU:0*
valueB?*    *r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Adam
VariableV2"/device:GPU:0*
shared_name *r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
dtype0*
	container *
shape:?
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Adam/AssignAssign?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Adam/Initializer/zeros"/device:GPU:0*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
validate_shape(*
use_locking(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Adam/readIdentity?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Adam"/device:GPU:0*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Adam_1/Initializer/zerosConst"/device:GPU:0*
valueB?*    *r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
dtype0
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Adam_1
VariableV2"/device:GPU:0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
dtype0*
	container *
shape:?*
shared_name 
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Adam_1/AssignAssign?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Adam_1?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
validate_shape(
?
?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Adam_1/readIdentity?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Adam_1"/device:GPU:0*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev
k
/main_level/agent/main/online/Adam/learning_rateConst"/device:GPU:0*
valueB
 *o?9*
dtype0
c
'main_level/agent/main/online/Adam/beta1Const"/device:GPU:0*
valueB
 *fff?*
dtype0
c
'main_level/agent/main/online/Adam/beta2Const"/device:GPU:0*
valueB
 *?p}?*
dtype0
e
)main_level/agent/main/online/Adam/epsilonConst"/device:GPU:0*
dtype0*
valueB
 *??8
?
?main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/ApplyAdam	ApplyAdamNmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_meanpmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adamrmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon%main_level/agent/main/online/0_holder"/device:GPU:0*
use_locking( *
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean*
use_nesterov( 
?
main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/ApplyAdam	ApplyAdamLmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_meannmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Adampmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon%main_level/agent/main/online/1_holder"/device:GPU:0*
use_locking( *
T0*_
_classU
SQloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean*
use_nesterov( 
?
?main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/ApplyAdam	ApplyAdamPmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddevrmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adamtmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon%main_level/agent/main/online/2_holder"/device:GPU:0*
use_locking( *
T0*c
_classY
WUloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev*
use_nesterov( 
?
?main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/ApplyAdam	ApplyAdamNmain_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddevpmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Adamrmain_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon%main_level/agent/main/online/3_holder"/device:GPU:0*
use_locking( *
T0*a
_classW
USloc:@main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev*
use_nesterov( 
?
{main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/ApplyAdam	ApplyAdamHmain_level/agent/main/online/network_0/gradients_from_head_0-0_rescalersjmain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adamlmain_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon%main_level/agent/main/online/4_holder"/device:GPU:0*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
use_nesterov( *
use_locking( 
?
?main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/ApplyAdam	ApplyAdamZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon%main_level/agent/main/online/5_holder"/device:GPU:0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
use_nesterov( *
use_locking( *
T0
?
?main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/ApplyAdam	ApplyAdamXmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_meanzmain_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Adam|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon%main_level/agent/main/online/6_holder"/device:GPU:0*
use_locking( *
T0*k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean*
use_nesterov( 
?
?main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/ApplyAdam	ApplyAdam\main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon%main_level/agent/main/online/7_holder"/device:GPU:0*
use_locking( *
T0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
use_nesterov( 
?
?main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/ApplyAdam	ApplyAdamZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Adam~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon%main_level/agent/main/online/8_holder"/device:GPU:0*
use_locking( *
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev*
use_nesterov( 
?
?main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/ApplyAdam	ApplyAdamZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon%main_level/agent/main/online/9_holder"/device:GPU:0*
use_locking( *
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean*
use_nesterov( 
?
?main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/ApplyAdam	ApplyAdamXmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_meanzmain_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Adam|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon&main_level/agent/main/online/10_holder"/device:GPU:0*
use_locking( *
T0*k
_classa
_]loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean*
use_nesterov( 
?
?main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/ApplyAdam	ApplyAdam\main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon&main_level/agent/main/online/11_holder"/device:GPU:0*
use_locking( *
T0*o
_classe
caloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
use_nesterov( 
?
?main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/ApplyAdam	ApplyAdamZmain_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev|main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Adam~main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon&main_level/agent/main/online/12_holder"/device:GPU:0*
use_locking( *
T0*m
_classc
a_loc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev*
use_nesterov( 
?
?main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/ApplyAdam	ApplyAdam_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon&main_level/agent/main/online/13_holder"/device:GPU:0*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
use_nesterov( *
use_locking( 
?
?main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/ApplyAdam	ApplyAdam]main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_meanmain_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon&main_level/agent/main/online/14_holder"/device:GPU:0*
use_locking( *
T0*p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean*
use_nesterov( 
?
?main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/ApplyAdam	ApplyAdamamain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon&main_level/agent/main/online/15_holder"/device:GPU:0*
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
use_nesterov( *
use_locking( 
?
?main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/ApplyAdam	ApplyAdam_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon&main_level/agent/main/online/16_holder"/device:GPU:0*
use_locking( *
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
use_nesterov( 
?
?main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/ApplyAdam	ApplyAdam_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon&main_level/agent/main/online/17_holder"/device:GPU:0*
use_locking( *
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
use_nesterov( 
?
?main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/ApplyAdam	ApplyAdam]main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_meanmain_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon&main_level/agent/main/online/18_holder"/device:GPU:0*
T0*p
_classf
dbloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean*
use_nesterov( *
use_locking( 
?
?main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/ApplyAdam	ApplyAdamamain_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon&main_level/agent/main/online/19_holder"/device:GPU:0*
use_locking( *
T0*t
_classj
hfloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
use_nesterov( 
?
?main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/ApplyAdam	ApplyAdam_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Adam?main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Adam_1-main_level/agent/main/online/beta1_power/read-main_level/agent/main/online/beta2_power/read/main_level/agent/main/online/Adam/learning_rate'main_level/agent/main/online/Adam/beta1'main_level/agent/main/online/Adam/beta2)main_level/agent/main/online/Adam/epsilon&main_level/agent/main/online/20_holder"/device:GPU:0*
T0*r
_classh
fdloc:@main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
use_nesterov( *
use_locking( 
?
%main_level/agent/main/online/Adam/mulMul-main_level/agent/main/online/beta1_power/read'main_level/agent/main/online/Adam/beta1|^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/ApplyAdam"/device:GPU:0*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers
?
(main_level/agent/main/online/Adam/AssignAssign(main_level/agent/main/online/beta1_power%main_level/agent/main/online/Adam/mul"/device:GPU:0*
use_locking( *
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
validate_shape(
?
'main_level/agent/main/online/Adam/mul_1Mul-main_level/agent/main/online/beta2_power/read'main_level/agent/main/online/Adam/beta2|^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/ApplyAdam"/device:GPU:0*
T0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers
?
*main_level/agent/main/online/Adam/Assign_1Assign(main_level/agent/main/online/beta2_power'main_level/agent/main/online/Adam/mul_1"/device:GPU:0*[
_classQ
OMloc:@main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers*
validate_shape(*
use_locking( *
T0
?
(main_level/agent/main/online/Adam/updateNoOp)^main_level/agent/main/online/Adam/Assign+^main_level/agent/main/online/Adam/Assign_1|^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/ApplyAdam?^main_level/agent/main/online/Adam/update_main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/ApplyAdam"/device:GPU:0
?
'main_level/agent/main/online/Adam/valueConst)^main_level/agent/main/online/Adam/update"/device:GPU:0*;
_class1
/-loc:@main_level/agent/main/online/global_step*
value	B	 R*
dtype0	
?
!main_level/agent/main/online/Adam	AssignAdd(main_level/agent/main/online/global_step'main_level/agent/main/online/Adam/value"/device:GPU:0*
use_locking( *
T0	*;
_class1
/-loc:@main_level/agent/main/online/global_step
e
,main_level/agent/main/online/AssignAdd/valueConst"/device:GPU:0*
dtype0	*
value	B	 R
?
&main_level/agent/main/online/AssignAdd	AssignAdd(main_level/agent/main/online/global_step,main_level/agent/main/online/AssignAdd/value"/device:GPU:0*
T0	*;
_class1
/-loc:@main_level/agent/main/online/global_step*
use_locking( 
?=
!main_level/agent/main/online/initNoOp0^main_level/agent/main/online/beta1_power/Assign0^main_level/agent/main/online/beta2_power/Assign0^main_level/agent/main/online/global_step/Assignr^main_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam/Assignt^main_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam_1/Assignv^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Adam/Assignx^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Adam_1/Assignx^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Adam/Assignz^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Adam_1/Assignx^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam/Assignz^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam_1/Assignz^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam/Assign|^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam_1/AssignP^main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/AssignT^main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/AssignV^main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/AssignV^main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/AssignX^main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Assigne^main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Assigng^main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Assigng^main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Assigni^main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Assigne^main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Assigng^main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Assigng^main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Assigni^main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Assign`^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Assignb^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Assignb^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Assignd^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Assign`^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Assignb^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Assignb^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Assignd^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Assign"/device:GPU:0
?
#main_level/agent/main/online/init_1NoOp-^main_level/agent/main/online/Variable/Assign7^main_level/agent/main/online/network_0/Variable/Assign"/device:GPU:0
?
'main_level/agent/main/online/group_depsNoOp"^main_level/agent/main/online/init$^main_level/agent/main/online/init_1"/device:GPU:0
l
3main_level/agent/main/target/Variable/initial_valueConst"/device:GPU:0*
value	B
 Z *
dtype0

?
%main_level/agent/main/target/Variable
VariableV2"/device:GPU:0*
dtype0
*
	container *
shape: *
shared_name 
?
,main_level/agent/main/target/Variable/AssignAssign%main_level/agent/main/target/Variable3main_level/agent/main/target/Variable/initial_value"/device:GPU:0*
use_locking(*
T0
*8
_class.
,*loc:@main_level/agent/main/target/Variable*
validate_shape(
?
*main_level/agent/main/target/Variable/readIdentity%main_level/agent/main/target/Variable"/device:GPU:0*
T0
*8
_class.
,*loc:@main_level/agent/main/target/Variable
b
(main_level/agent/main/target/PlaceholderPlaceholder"/device:GPU:0*
shape:*
dtype0

?
#main_level/agent/main/target/AssignAssign%main_level/agent/main/target/Variable(main_level/agent/main/target/Placeholder"/device:GPU:0*
use_locking(*
T0
*8
_class.
,*loc:@main_level/agent/main/target/Variable*
validate_shape(
?
>main_level/agent/main/target/network_0/observation/observationPlaceholder"/device:GPU:0*
dtype0*
shape:??????????
x
<main_level/agent/main/target/network_0/observation/truediv/yConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
:main_level/agent/main/target/network_0/observation/truedivRealDiv>main_level/agent/main/target/network_0/observation/observation<main_level/agent/main/target/network_0/observation/truediv/y"/device:GPU:0*
T0
t
8main_level/agent/main/target/network_0/observation/sub/yConst"/device:GPU:0*
valueB
 *    *
dtype0
?
6main_level/agent/main/target/network_0/observation/subSub:main_level/agent/main/target/network_0/observation/truediv8main_level/agent/main/target/network_0/observation/sub/y"/device:GPU:0*
T0
?
omain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/shapeConst*a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean*
valueB"H     *
dtype0
?
mmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/minConst*a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean*
valueB
 *OS?*
dtype0
?
mmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/maxConst*a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean*
valueB
 *OS=*
dtype0
?
wmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/RandomUniformRandomUniformomain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/shape*
T0*a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean*
dtype0*
seed2 *

seed 
?
mmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/subSubmmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/maxmmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/min*
T0*a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean
?
mmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/mulMulwmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/RandomUniformmmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/sub*
T0*a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean
?
imain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniformAddmmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/mulmmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform/min*
T0*a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean
?
Nmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean
VariableV2"/device:GPU:0*a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean*
dtype0*
	container *
shape:
??*
shared_name 
?
Umain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean/AssignAssignNmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_meanimain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean/Initializer/random_uniform"/device:GPU:0*a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean*
validate_shape(*
use_locking(*
T0
?
Smain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean/readIdentityNmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean"/device:GPU:0*
T0*a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean
?
^main_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_mean/Initializer/zerosConst*_
_classU
SQloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_mean*
valueB?*    *
dtype0
?
Lmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_mean
VariableV2"/device:GPU:0*
shared_name *_
_classU
SQloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_mean*
dtype0*
	container *
shape:?
?
Smain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_mean/AssignAssignLmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_mean^main_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_mean/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*_
_classU
SQloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_mean*
validate_shape(
?
Qmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_mean/readIdentityLmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_mean"/device:GPU:0*
T0*_
_classU
SQloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_mean
?
qmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/shapeConst*c
_classY
WUloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev*
valueB"H     *
dtype0
?
omain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/minConst*c
_classY
WUloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev*
valueB
 *OS??*
dtype0
?
omain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/maxConst*
dtype0*c
_classY
WUloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev*
valueB
 *OS?<
?
ymain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/RandomUniformRandomUniformqmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/shape*

seed *
T0*c
_classY
WUloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev*
dtype0*
seed2 
?
omain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/subSubomain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/maxomain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/min*
T0*c
_classY
WUloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev
?
omain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/mulMulymain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/RandomUniformomain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/sub*c
_classY
WUloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev*
T0
?
kmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniformAddomain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/mulomain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform/min*
T0*c
_classY
WUloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev
?
Pmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev
VariableV2"/device:GPU:0*
shared_name *c
_classY
WUloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev*
dtype0*
	container *
shape:
??
?
Wmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev/AssignAssignPmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddevkmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*c
_classY
WUloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev*
validate_shape(
?
Umain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev/readIdentityPmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev"/device:GPU:0*
T0*c
_classY
WUloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev
?
omain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/shapeConst*a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev*
valueB:?*
dtype0
?
mmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/minConst*a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev*
valueB
 *OS??*
dtype0
?
mmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/maxConst*a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev*
valueB
 *OS?<*
dtype0
?
wmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/RandomUniformRandomUniformomain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/shape*a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev*
dtype0*
seed2 *

seed *
T0
?
mmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/subSubmmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/maxmmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/min*
T0*a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev
?
mmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/mulMulwmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/RandomUniformmmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/sub*
T0*a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev
?
imain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniformAddmmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/mulmmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform/min*
T0*a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev
?
Nmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev
VariableV2"/device:GPU:0*
shared_name *a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev*
dtype0*
	container *
shape:?
?
Umain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev/AssignAssignNmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddevimain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev*
validate_shape(
?
Smain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev/readIdentityNmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev"/device:GPU:0*
T0*a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev
?
Vmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal/shapeConst"/device:GPU:0*
valueB:?*
dtype0
?
Umain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
Wmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
emain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal/RandomStandardNormalRandomStandardNormalVmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal/shape"/device:GPU:0*
dtype0*
seed2 *

seed *
T0
?
Tmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal/mulMulemain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal/RandomStandardNormalWmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal/stddev"/device:GPU:0*
T0
?
Pmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normalAddTmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal/mulUmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal/mean"/device:GPU:0*
T0
?
Fmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/AbsAbsPmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal"/device:GPU:0*
T0
?
Gmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/SqrtSqrtFmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/Abs"/device:GPU:0*
T0
?
Gmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/SignSignPmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal"/device:GPU:0*
T0
?
Fmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/mulMulGmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/SqrtGmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/Sign"/device:GPU:0*
T0
?
Xmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_1/shapeConst"/device:GPU:0*
valueB"H     *
dtype0
?
Wmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_1/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
Ymain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_1/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
gmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_1/RandomStandardNormalRandomStandardNormalXmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_1/shape"/device:GPU:0*

seed *
T0*
dtype0*
seed2 
?
Vmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_1/mulMulgmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_1/RandomStandardNormalYmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_1/stddev"/device:GPU:0*
T0
?
Rmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_1AddVmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_1/mulWmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_1/mean"/device:GPU:0*
T0
?
Hmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/Abs_1AbsRmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_1"/device:GPU:0*
T0
?
Imain_level/agent/main/target/network_0/observation/NoisyNetDense_0/Sqrt_1SqrtHmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/Abs_1"/device:GPU:0*
T0
?
Imain_level/agent/main/target/network_0/observation/NoisyNetDense_0/Sign_1SignRmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_1"/device:GPU:0*
T0
?
Hmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/mul_1MulImain_level/agent/main/target/network_0/observation/NoisyNetDense_0/Sqrt_1Imain_level/agent/main/target/network_0/observation/NoisyNetDense_0/Sign_1"/device:GPU:0*
T0
?
Xmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_2/shapeConst"/device:GPU:0*
dtype0*
valueB"      
?
Wmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_2/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
Ymain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_2/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
gmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_2/RandomStandardNormalRandomStandardNormalXmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_2/shape"/device:GPU:0*

seed *
T0*
dtype0*
seed2 
?
Vmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_2/mulMulgmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_2/RandomStandardNormalYmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_2/stddev"/device:GPU:0*
T0
?
Rmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_2AddVmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_2/mulWmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_2/mean"/device:GPU:0*
T0
?
Hmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/Abs_2AbsRmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_2"/device:GPU:0*
T0
?
Imain_level/agent/main/target/network_0/observation/NoisyNetDense_0/Sqrt_2SqrtHmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/Abs_2"/device:GPU:0*
T0
?
Imain_level/agent/main/target/network_0/observation/NoisyNetDense_0/Sign_2SignRmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/random_normal_2"/device:GPU:0*
T0
?
Hmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/mul_2MulImain_level/agent/main/target/network_0/observation/NoisyNetDense_0/Sqrt_2Imain_level/agent/main/target/network_0/observation/NoisyNetDense_0/Sign_2"/device:GPU:0*
T0
?
Imain_level/agent/main/target/network_0/observation/NoisyNetDense_0/MatMulMatMulHmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/mul_1Hmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/mul_2"/device:GPU:0*
T0*
transpose_a( *
transpose_b( 
?
6main_level/agent/main/target/network_0/observation/mulMulSmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev/readFmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/mul"/device:GPU:0*
T0
?
6main_level/agent/main/target/network_0/observation/addAddQmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_mean/read6main_level/agent/main/target/network_0/observation/mul"/device:GPU:0*
T0
?
8main_level/agent/main/target/network_0/observation/mul_1MulUmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev/readImain_level/agent/main/target/network_0/observation/NoisyNetDense_0/MatMul"/device:GPU:0*
T0
?
8main_level/agent/main/target/network_0/observation/add_1AddSmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean/read8main_level/agent/main/target/network_0/observation/mul_1"/device:GPU:0*
T0
?
9main_level/agent/main/target/network_0/observation/MatMulMatMul6main_level/agent/main/target/network_0/observation/sub8main_level/agent/main/target/network_0/observation/add_1"/device:GPU:0*
transpose_a( *
transpose_b( *
T0
?
8main_level/agent/main/target/network_0/observation/add_2Add9main_level/agent/main/target/network_0/observation/MatMul6main_level/agent/main/target/network_0/observation/add"/device:GPU:0*
T0
?
Zmain_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activationRelu8main_level/agent/main/target/network_0/observation/add_2"/device:GPU:0*
T0
?
Hmain_level/agent/main/target/network_0/observation/Flatten/flatten/ShapeShapeZmain_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0
?
Vmain_level/agent/main/target/network_0/observation/Flatten/flatten/strided_slice/stackConst"/device:GPU:0*
valueB: *
dtype0
?
Xmain_level/agent/main/target/network_0/observation/Flatten/flatten/strided_slice/stack_1Const"/device:GPU:0*
dtype0*
valueB:
?
Xmain_level/agent/main/target/network_0/observation/Flatten/flatten/strided_slice/stack_2Const"/device:GPU:0*
valueB:*
dtype0
?
Pmain_level/agent/main/target/network_0/observation/Flatten/flatten/strided_sliceStridedSliceHmain_level/agent/main/target/network_0/observation/Flatten/flatten/ShapeVmain_level/agent/main/target/network_0/observation/Flatten/flatten/strided_slice/stackXmain_level/agent/main/target/network_0/observation/Flatten/flatten/strided_slice/stack_1Xmain_level/agent/main/target/network_0/observation/Flatten/flatten/strided_slice/stack_2"/device:GPU:0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
?
Rmain_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape/shape/1Const"/device:GPU:0*
valueB :
?????????*
dtype0
?
Pmain_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape/shapePackPmain_level/agent/main/target/network_0/observation/Flatten/flatten/strided_sliceRmain_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape/shape/1"/device:GPU:0*
T0*

axis *
N
?
Jmain_level/agent/main/target/network_0/observation/Flatten/flatten/ReshapeReshapeZmain_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activationPmain_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape/shape"/device:GPU:0*
T0*
Tshape0
}
=main_level/agent/main/target/network_0/Variable/initial_valueConst"/device:GPU:0*
dtype0*
valueB*  ??
?
/main_level/agent/main/target/network_0/Variable
VariableV2"/device:GPU:0*
shape:*
shared_name *
dtype0*
	container 
?
6main_level/agent/main/target/network_0/Variable/AssignAssign/main_level/agent/main/target/network_0/Variable=main_level/agent/main/target/network_0/Variable/initial_value"/device:GPU:0*
use_locking(*
T0*B
_class8
64loc:@main_level/agent/main/target/network_0/Variable*
validate_shape(
?
4main_level/agent/main/target/network_0/Variable/readIdentity/main_level/agent/main/target/network_0/Variable"/device:GPU:0*
T0*B
_class8
64loc:@main_level/agent/main/target/network_0/Variable
?
,main_level/agent/main/target/network_0/ConstConst"/device:GPU:0*
dtype0*?
value?B?3"?   ????33????ff?   ?33??ff??????????  ??33??ff???????̌?  ??fff???L?333????   ???̿??????L???̾    ???>??L?????????   @??@333@??L@fff@  ?@?̌@???@ff?@33?@  ?@???@???@ff?@33?@   AffA??A33A??A   A
?
+main_level/agent/main/target/network_0/CastCast,main_level/agent/main/target/network_0/Const"/device:GPU:0*

SrcT0*
Truncate( *

DstT0
C
Const_1Const"/device:GPU:0*
valueB
 *  ??*
dtype0
?
Vmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers/initial_valueConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
Hmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers
VariableV2"/device:GPU:0*
shared_name *
dtype0*
	container *
shape: 
?
Omain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers/AssignAssignHmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalersVmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers/initial_value"/device:GPU:0*
use_locking(*
T0*[
_classQ
OMloc:@main_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers*
validate_shape(
?
Mmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers/readIdentityHmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers"/device:GPU:0*
T0*[
_classQ
OMloc:@main_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers
?
Jmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers_1Placeholder"/device:GPU:0*
shape:*
dtype0
?
-main_level/agent/main/target/network_0/AssignAssignHmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalersJmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers_1"/device:GPU:0*
use_locking( *
T0*[
_classQ
OMloc:@main_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers*
validate_shape(
h
,main_level/agent/main/target/network_0/sub/xConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
*main_level/agent/main/target/network_0/subSub,main_level/agent/main/target/network_0/sub/xMmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers/read"/device:GPU:0*
T0
?
9main_level/agent/main/target/network_0/StopGradient/inputPackJmain_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape"/device:GPU:0*
T0*

axis *
N
?
3main_level/agent/main/target/network_0/StopGradientStopGradient9main_level/agent/main/target/network_0/StopGradient/input"/device:GPU:0*
T0
?
*main_level/agent/main/target/network_0/mulMul*main_level/agent/main/target/network_0/sub3main_level/agent/main/target/network_0/StopGradient"/device:GPU:0*
T0
?
.main_level/agent/main/target/network_0/mul_1/yPackJmain_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape"/device:GPU:0*
T0*

axis *
N
?
,main_level/agent/main/target/network_0/mul_1MulMmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers/read.main_level/agent/main/target/network_0/mul_1/y"/device:GPU:0*
T0
?
*main_level/agent/main/target/network_0/addAdd*main_level/agent/main/target/network_0/mul,main_level/agent/main/target/network_0/mul_1"/device:GPU:0*
T0
?
Rmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice/stackConst"/device:GPU:0*
valueB: *
dtype0
?
Tmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice/stack_1Const"/device:GPU:0*
valueB:*
dtype0
?
Tmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice/stack_2Const"/device:GPU:0*
valueB:*
dtype0
?
Lmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_sliceStridedSlice*main_level/agent/main/target/network_0/addRmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice/stackTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice/stack_1Tmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice/stack_2"/device:GPU:0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
?
{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/shapeConst*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
valueB"      *
dtype0
?
ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/minConst*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
valueB
 *  ??*
dtype0
?
ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/maxConst*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
valueB
 *  ?=*
dtype0
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/RandomUniformRandomUniform{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/shape*
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
dtype0*
seed2 *

seed 
?
ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/subSubymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/maxymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/min*
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean
?
ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/mulMul?main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/RandomUniformymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/sub*
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean
?
umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniformAddymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/mulymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform/min*
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean
?
Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean
VariableV2"/device:GPU:0*
shape:
??*
shared_name *m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
dtype0*
	container 
?
amain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/AssignAssignZmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_meanumain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
validate_shape(
?
_main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/readIdentityZmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean"/device:GPU:0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
T0
?
jmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Initializer/zerosConst*k
_classa
_]loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean*
valueB?*    *
dtype0
?
Xmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean
VariableV2"/device:GPU:0*
	container *
shape:?*
shared_name *k
_classa
_]loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean*
dtype0
?
_main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/AssignAssignXmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_meanjmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Initializer/zeros"/device:GPU:0*k
_classa
_]loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean*
validate_shape(*
use_locking(*
T0
?
]main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/readIdentityXmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean"/device:GPU:0*
T0*k
_classa
_]loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean
?
}main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/shapeConst*o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
valueB"      *
dtype0
?
{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/minConst*o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
valueB
 *   ?*
dtype0
?
{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/maxConst*o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
valueB
 *   =*
dtype0
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/RandomUniformRandomUniform}main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev
?
{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/subSub{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/max{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/min*
T0*o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev
?
{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/mulMul?main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/RandomUniform{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/sub*
T0*o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev
?
wmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniformAdd{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/mul{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform/min*o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
T0
?
\main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev
VariableV2"/device:GPU:0*
	container *
shape:
??*
shared_name *o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
dtype0
?
cmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/AssignAssign\main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddevwmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
validate_shape(
?
amain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/readIdentity\main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev"/device:GPU:0*
T0*o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev
?
{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/shapeConst*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev*
valueB:?*
dtype0
?
ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/minConst*
dtype0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev*
valueB
 *   ?
?
ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/maxConst*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev*
valueB
 *   =*
dtype0
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/RandomUniformRandomUniform{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/shape*
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev*
dtype0*
seed2 *

seed 
?
ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/subSubymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/maxymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/min*
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev
?
ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/mulMul?main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/RandomUniformymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/sub*
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev
?
umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniformAddymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/mulymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform/min*
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev
?
Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev
VariableV2"/device:GPU:0*
dtype0*
	container *
shape:?*
shared_name *m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev
?
amain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/AssignAssignZmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddevumain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Initializer/random_uniform"/device:GPU:0*
validate_shape(*
use_locking(*
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev
?
_main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/readIdentityZmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev"/device:GPU:0*
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev
?
bmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal/shapeConst"/device:GPU:0*
valueB:?*
dtype0
?
amain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
cmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
qmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal/RandomStandardNormalRandomStandardNormalbmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal/shape"/device:GPU:0*
T0*
dtype0*
seed2 *

seed 
?
`main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal/mulMulqmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal/RandomStandardNormalcmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal/stddev"/device:GPU:0*
T0
?
\main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normalAdd`main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal/mulamain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal/mean"/device:GPU:0*
T0
?
Rmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/AbsAbs\main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal"/device:GPU:0*
T0
?
Smain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/SqrtSqrtRmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/Abs"/device:GPU:0*
T0
?
Smain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/SignSign\main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal"/device:GPU:0*
T0
?
Rmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/mulMulSmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/SqrtSmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/Sign"/device:GPU:0*
T0
?
dmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1/shapeConst"/device:GPU:0*
valueB"      *
dtype0
?
cmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
emain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
smain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1/RandomStandardNormalRandomStandardNormaldmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1/shape"/device:GPU:0*
T0*
dtype0*
seed2 *

seed 
?
bmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1/mulMulsmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1/RandomStandardNormalemain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1/stddev"/device:GPU:0*
T0
?
^main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1Addbmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1/mulcmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1/mean"/device:GPU:0*
T0
?
Tmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/Abs_1Abs^main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1"/device:GPU:0*
T0
?
Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/Sqrt_1SqrtTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/Abs_1"/device:GPU:0*
T0
?
Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/Sign_1Sign^main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_1"/device:GPU:0*
T0
?
Tmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/mul_1MulUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/Sqrt_1Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/Sign_1"/device:GPU:0*
T0
?
dmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2/shapeConst"/device:GPU:0*
valueB"      *
dtype0
?
cmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
emain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
smain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2/RandomStandardNormalRandomStandardNormaldmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2/shape"/device:GPU:0*
seed2 *

seed *
T0*
dtype0
?
bmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2/mulMulsmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2/RandomStandardNormalemain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2/stddev"/device:GPU:0*
T0
?
^main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2Addbmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2/mulcmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2/mean"/device:GPU:0*
T0
?
Tmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/Abs_2Abs^main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2"/device:GPU:0*
T0
?
Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/Sqrt_2SqrtTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/Abs_2"/device:GPU:0*
T0
?
Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/Sign_2Sign^main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/random_normal_2"/device:GPU:0*
T0
?
Tmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/mul_2MulUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/Sqrt_2Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/Sign_2"/device:GPU:0*
T0
?
Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/MatMulMatMulTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/mul_1Tmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/mul_2"/device:GPU:0*
transpose_a( *
transpose_b( *
T0
?
Nmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mulMul_main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/readRmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/mul"/device:GPU:0*
T0
?
Nmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/addAdd]main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/readNmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul"/device:GPU:0*
T0
?
Pmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_1Mulamain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/readUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/MatMul"/device:GPU:0*
T0
?
Pmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_1Add_main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/readPmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_1"/device:GPU:0*
T0
?
Qmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMulMatMulLmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slicePmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_1"/device:GPU:0*
T0*
transpose_a( *
transpose_b( 
?
Pmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2AddQmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMulNmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add"/device:GPU:0*
T0
?
Omain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ReluReluPmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2"/device:GPU:0*
T0
?
{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/shapeConst*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean*
valueB"   3   *
dtype0
?
ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/minConst*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean*
valueB
 *?5?*
dtype0
?
ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/maxConst*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean*
valueB
 *?5=*
dtype0
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/RandomUniformRandomUniform{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean
?
ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/subSubymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/maxymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/min*
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean
?
ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/mulMul?main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/RandomUniformymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/sub*
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean
?
umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniformAddymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/mulymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform/min*
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean
?
Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean
VariableV2"/device:GPU:0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean*
dtype0*
	container *
shape:	?3*
shared_name 
?
amain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/AssignAssignZmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_meanumain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean*
validate_shape(
?
_main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/readIdentityZmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean"/device:GPU:0*
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean
?
jmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Initializer/zerosConst*k
_classa
_]loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean*
valueB3*    *
dtype0
?
Xmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean
VariableV2"/device:GPU:0*
shape:3*
shared_name *k
_classa
_]loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean*
dtype0*
	container 
?
_main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/AssignAssignXmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_meanjmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*k
_classa
_]loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean*
validate_shape(
?
]main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/readIdentityXmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean"/device:GPU:0*
T0*k
_classa
_]loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean
?
}main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/shapeConst*o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
valueB"   3   *
dtype0
?
{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/minConst*o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
valueB
 *???*
dtype0
?
{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/maxConst*o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
valueB
 *??<*
dtype0
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/RandomUniformRandomUniform}main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev
?
{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/subSub{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/max{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/min*
T0*o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev
?
{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/mulMul?main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/RandomUniform{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/sub*
T0*o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev
?
wmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniformAdd{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/mul{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform/min*
T0*o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev
?
\main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev
VariableV2"/device:GPU:0*
shape:	?3*
shared_name *o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
dtype0*
	container 
?
cmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/AssignAssign\main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddevwmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Initializer/random_uniform"/device:GPU:0*
T0*o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
validate_shape(*
use_locking(
?
amain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/readIdentity\main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev"/device:GPU:0*
T0*o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev
?
{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/shapeConst*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev*
valueB:3*
dtype0
?
ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/minConst*
dtype0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev*
valueB
 *???
?
ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/maxConst*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev*
valueB
 *??<*
dtype0
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/RandomUniformRandomUniform{main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/shape*
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev*
dtype0*
seed2 *

seed 
?
ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/subSubymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/maxymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/min*
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev
?
ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/mulMul?main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/RandomUniformymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/sub*
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev
?
umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniformAddymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/mulymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform/min*
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev
?
Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev
VariableV2"/device:GPU:0*
dtype0*
	container *
shape:3*
shared_name *m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev
?
amain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/AssignAssignZmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddevumain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Initializer/random_uniform"/device:GPU:0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev*
validate_shape(*
use_locking(*
T0
?
_main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/readIdentityZmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev"/device:GPU:0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev*
T0
?
bmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal/shapeConst"/device:GPU:0*
valueB:3*
dtype0
?
amain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
cmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
qmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal/RandomStandardNormalRandomStandardNormalbmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal/shape"/device:GPU:0*
T0*
dtype0*
seed2 *

seed 
?
`main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal/mulMulqmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal/RandomStandardNormalcmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal/stddev"/device:GPU:0*
T0
?
\main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normalAdd`main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal/mulamain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal/mean"/device:GPU:0*
T0
?
Rmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/AbsAbs\main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal"/device:GPU:0*
T0
?
Smain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/SqrtSqrtRmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/Abs"/device:GPU:0*
T0
?
Smain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/SignSign\main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal"/device:GPU:0*
T0
?
Rmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/mulMulSmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/SqrtSmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/Sign"/device:GPU:0*
T0
?
dmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1/shapeConst"/device:GPU:0*
valueB"      *
dtype0
?
cmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
emain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1/stddevConst"/device:GPU:0*
dtype0*
valueB
 *  ??
?
smain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1/RandomStandardNormalRandomStandardNormaldmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1/shape"/device:GPU:0*
seed2 *

seed *
T0*
dtype0
?
bmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1/mulMulsmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1/RandomStandardNormalemain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1/stddev"/device:GPU:0*
T0
?
^main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1Addbmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1/mulcmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1/mean"/device:GPU:0*
T0
?
Tmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/Abs_1Abs^main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1"/device:GPU:0*
T0
?
Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/Sqrt_1SqrtTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/Abs_1"/device:GPU:0*
T0
?
Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/Sign_1Sign^main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_1"/device:GPU:0*
T0
?
Tmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/mul_1MulUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/Sqrt_1Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/Sign_1"/device:GPU:0*
T0
?
dmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2/shapeConst"/device:GPU:0*
valueB"   3   *
dtype0
?
cmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
emain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2/stddevConst"/device:GPU:0*
dtype0*
valueB
 *  ??
?
smain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2/RandomStandardNormalRandomStandardNormaldmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2/shape"/device:GPU:0*

seed *
T0*
dtype0*
seed2 
?
bmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2/mulMulsmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2/RandomStandardNormalemain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2/stddev"/device:GPU:0*
T0
?
^main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2Addbmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2/mulcmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2/mean"/device:GPU:0*
T0
?
Tmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/Abs_2Abs^main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2"/device:GPU:0*
T0
?
Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/Sqrt_2SqrtTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/Abs_2"/device:GPU:0*
T0
?
Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/Sign_2Sign^main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/random_normal_2"/device:GPU:0*
T0
?
Tmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/mul_2MulUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/Sqrt_2Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/Sign_2"/device:GPU:0*
T0
?
Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/MatMulMatMulTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/mul_1Tmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/mul_2"/device:GPU:0*
transpose_b( *
T0*
transpose_a( 
?
Pmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_2Mul_main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/readRmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/mul"/device:GPU:0*
T0
?
Pmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_3Add]main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/readPmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_2"/device:GPU:0*
T0
?
Pmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_3Mulamain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/readUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/MatMul"/device:GPU:0*
T0
?
Pmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_4Add_main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/readPmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_3"/device:GPU:0*
T0
?
Smain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_1MatMulOmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ReluPmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_4"/device:GPU:0*
transpose_a( *
transpose_b( *
T0
?
Pmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5AddSmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_1Pmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_3"/device:GPU:0*
T0
?
Ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDims/dimConst"/device:GPU:0*
value	B :*
dtype0
?
Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDims
ExpandDimsPmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5Ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDims/dim"/device:GPU:0*

Tdim0*
T0
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/shapeConst*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
valueB"      *
dtype0
?
~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/minConst*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
valueB
 *  ??*
dtype0
?
~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/maxConst*
dtype0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
valueB
 *  ?=
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/RandomUniformRandomUniform?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/shape*

seed *
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
dtype0*
seed2 
?
~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/subSub~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/max~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/min*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
T0
?
~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/mulMul?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/RandomUniform~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/sub*
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean
?
zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniformAdd~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/mul~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform/min*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
T0
?
_main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean
VariableV2"/device:GPU:0*
shape:
??*
shared_name *r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
dtype0*
	container 
?
fmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/AssignAssign_main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_meanzmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Initializer/random_uniform"/device:GPU:0*
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
validate_shape(*
use_locking(
?
dmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/readIdentity_main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean"/device:GPU:0*
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean
?
omain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Initializer/zerosConst*p
_classf
dbloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean*
valueB?*    *
dtype0
?
]main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean
VariableV2"/device:GPU:0*
shape:?*
shared_name *p
_classf
dbloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean*
dtype0*
	container 
?
dmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/AssignAssign]main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_meanomain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*p
_classf
dbloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean*
validate_shape(
?
bmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/readIdentity]main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean"/device:GPU:0*
T0*p
_classf
dbloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/shapeConst*t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
valueB"      *
dtype0
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/minConst*t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
valueB
 *   ?*
dtype0
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/maxConst*t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
valueB
 *   =*
dtype0
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/RandomUniformRandomUniform?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/shape*
T0*t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
dtype0*
seed2 *

seed 
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/subSub?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/max?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/min*t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
T0
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/mulMul?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/RandomUniform?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/sub*
T0*t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev
?
|main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniformAdd?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/mul?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform/min*
T0*t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev
?
amain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev
VariableV2"/device:GPU:0*
shape:
??*
shared_name *t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
dtype0*
	container 
?
hmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/AssignAssignamain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev|main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Initializer/random_uniform"/device:GPU:0*
validate_shape(*
use_locking(*
T0*t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev
?
fmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/readIdentityamain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev"/device:GPU:0*
T0*t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/shapeConst*
dtype0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
valueB:?
?
~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/minConst*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
valueB
 *   ?*
dtype0
?
~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/maxConst*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
valueB
 *   =*
dtype0
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/RandomUniformRandomUniform?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/shape*
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
dtype0*
seed2 *

seed 
?
~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/subSub~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/max~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/min*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
T0
?
~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/mulMul?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/RandomUniform~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/sub*
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev
?
zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniformAdd~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/mul~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform/min*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
T0
?
_main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev
VariableV2"/device:GPU:0*
	container *
shape:?*
shared_name *r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
dtype0
?
fmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/AssignAssign_main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddevzmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Initializer/random_uniform"/device:GPU:0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
validate_shape(*
use_locking(*
T0
?
dmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/readIdentity_main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev"/device:GPU:0*
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev
?
gmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal/shapeConst"/device:GPU:0*
valueB:?*
dtype0
?
fmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal/meanConst"/device:GPU:0*
dtype0*
valueB
 *    
?
hmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
vmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal/RandomStandardNormalRandomStandardNormalgmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal/shape"/device:GPU:0*
T0*
dtype0*
seed2 *

seed 
?
emain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal/mulMulvmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal/RandomStandardNormalhmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal/stddev"/device:GPU:0*
T0
?
amain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normalAddemain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal/mulfmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal/mean"/device:GPU:0*
T0
?
Wmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/AbsAbsamain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal"/device:GPU:0*
T0
?
Xmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/SqrtSqrtWmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/Abs"/device:GPU:0*
T0
?
Xmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/SignSignamain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal"/device:GPU:0*
T0
?
Wmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/mulMulXmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/SqrtXmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/Sign"/device:GPU:0*
T0
?
imain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1/shapeConst"/device:GPU:0*
valueB"      *
dtype0
?
hmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
jmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
xmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1/RandomStandardNormalRandomStandardNormalimain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1/shape"/device:GPU:0*
dtype0*
seed2 *

seed *
T0
?
gmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1/mulMulxmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1/RandomStandardNormaljmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1/stddev"/device:GPU:0*
T0
?
cmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1Addgmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1/mulhmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1/mean"/device:GPU:0*
T0
?
Ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/Abs_1Abscmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1"/device:GPU:0*
T0
?
Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/Sqrt_1SqrtYmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/Abs_1"/device:GPU:0*
T0
?
Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/Sign_1Signcmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_1"/device:GPU:0*
T0
?
Ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/mul_1MulZmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/Sqrt_1Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/Sign_1"/device:GPU:0*
T0
?
imain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2/shapeConst"/device:GPU:0*
dtype0*
valueB"      
?
hmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
jmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
xmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2/RandomStandardNormalRandomStandardNormalimain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2/shape"/device:GPU:0*

seed *
T0*
dtype0*
seed2 
?
gmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2/mulMulxmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2/RandomStandardNormaljmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2/stddev"/device:GPU:0*
T0
?
cmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2Addgmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2/mulhmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2/mean"/device:GPU:0*
T0
?
Ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/Abs_2Abscmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2"/device:GPU:0*
T0
?
Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/Sqrt_2SqrtYmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/Abs_2"/device:GPU:0*
T0
?
Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/Sign_2Signcmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/random_normal_2"/device:GPU:0*
T0
?
Ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/mul_2MulZmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/Sqrt_2Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/Sign_2"/device:GPU:0*
T0
?
Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/MatMulMatMulYmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/mul_1Ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/mul_2"/device:GPU:0*
transpose_a( *
transpose_b( *
T0
?
Smain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mulMuldmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/readWmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/mul"/device:GPU:0*
T0
?
Smain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/addAddbmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/readSmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul"/device:GPU:0*
T0
?
Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_1Mulfmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/readZmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/MatMul"/device:GPU:0*
T0
?
Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_1Adddmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/readUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_1"/device:GPU:0*
T0
?
Vmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMulMatMulLmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_sliceUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_1"/device:GPU:0*
T0*
transpose_a( *
transpose_b( 
?
Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2AddVmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMulSmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add"/device:GPU:0*
T0
?
Tmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/ReluReluUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2"/device:GPU:0*
T0
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/shapeConst*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
valueB"   ?   *
dtype0
?
~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/minConst*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
valueB
 *?5?*
dtype0
?
~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/maxConst*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
valueB
 *?5=*
dtype0
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/RandomUniformRandomUniform?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean
?
~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/subSub~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/max~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/min*
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean
?
~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/mulMul?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/RandomUniform~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/sub*
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean
?
zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniformAdd~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/mul~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform/min*
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean
?
_main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean
VariableV2"/device:GPU:0*
	container *
shape:
??*
shared_name *r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
dtype0
?
fmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/AssignAssign_main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_meanzmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Initializer/random_uniform"/device:GPU:0*
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
validate_shape(*
use_locking(
?
dmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/readIdentity_main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean"/device:GPU:0*
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean
?
omain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Initializer/zerosConst*p
_classf
dbloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean*
valueB?*    *
dtype0
?
]main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean
VariableV2"/device:GPU:0*
shape:?*
shared_name *p
_classf
dbloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean*
dtype0*
	container 
?
dmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/AssignAssign]main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_meanomain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*p
_classf
dbloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean*
validate_shape(
?
bmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/readIdentity]main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean"/device:GPU:0*
T0*p
_classf
dbloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/shapeConst*t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
valueB"   ?   *
dtype0
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/minConst*t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
valueB
 *???*
dtype0
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/maxConst*t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
valueB
 *??<*
dtype0
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/RandomUniformRandomUniform?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/shape*
T0*t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
dtype0*
seed2 *

seed 
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/subSub?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/max?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/min*
T0*t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/mulMul?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/RandomUniform?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/sub*
T0*t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev
?
|main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniformAdd?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/mul?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform/min*
T0*t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev
?
amain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev
VariableV2"/device:GPU:0*t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
dtype0*
	container *
shape:
??*
shared_name 
?
hmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/AssignAssignamain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev|main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Initializer/random_uniform"/device:GPU:0*
use_locking(*
T0*t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
validate_shape(
?
fmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/readIdentityamain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev"/device:GPU:0*
T0*t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/shapeConst*
dtype0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
valueB:?
?
~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/minConst*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
valueB
 *???*
dtype0
?
~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/maxConst*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
valueB
 *??<*
dtype0
?
?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/RandomUniformRandomUniform?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/shape*
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
dtype0*
seed2 *

seed 
?
~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/subSub~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/max~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/min*
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev
?
~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/mulMul?main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/RandomUniform~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/sub*
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev
?
zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniformAdd~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/mul~main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform/min*
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev
?
_main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev
VariableV2"/device:GPU:0*
shared_name *r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
dtype0*
	container *
shape:?
?
fmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/AssignAssign_main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddevzmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Initializer/random_uniform"/device:GPU:0*
validate_shape(*
use_locking(*
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev
?
dmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/readIdentity_main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev"/device:GPU:0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
T0
?
gmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal/shapeConst"/device:GPU:0*
valueB:?*
dtype0
?
fmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
hmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
vmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal/RandomStandardNormalRandomStandardNormalgmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal/shape"/device:GPU:0*
seed2 *

seed *
T0*
dtype0
?
emain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal/mulMulvmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal/RandomStandardNormalhmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal/stddev"/device:GPU:0*
T0
?
amain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normalAddemain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal/mulfmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal/mean"/device:GPU:0*
T0
?
Wmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/AbsAbsamain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal"/device:GPU:0*
T0
?
Xmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/SqrtSqrtWmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/Abs"/device:GPU:0*
T0
?
Xmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/SignSignamain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal"/device:GPU:0*
T0
?
Wmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/mulMulXmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/SqrtXmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/Sign"/device:GPU:0*
T0
?
imain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1/shapeConst"/device:GPU:0*
valueB"      *
dtype0
?
hmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
jmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
xmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1/RandomStandardNormalRandomStandardNormalimain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1/shape"/device:GPU:0*
T0*
dtype0*
seed2 *

seed 
?
gmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1/mulMulxmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1/RandomStandardNormaljmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1/stddev"/device:GPU:0*
T0
?
cmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1Addgmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1/mulhmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1/mean"/device:GPU:0*
T0
?
Ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/Abs_1Abscmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1"/device:GPU:0*
T0
?
Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/Sqrt_1SqrtYmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/Abs_1"/device:GPU:0*
T0
?
Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/Sign_1Signcmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_1"/device:GPU:0*
T0
?
Ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/mul_1MulZmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/Sqrt_1Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/Sign_1"/device:GPU:0*
T0
?
imain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2/shapeConst"/device:GPU:0*
valueB"   ?   *
dtype0
?
hmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2/meanConst"/device:GPU:0*
valueB
 *    *
dtype0
?
jmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2/stddevConst"/device:GPU:0*
valueB
 *  ??*
dtype0
?
xmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2/RandomStandardNormalRandomStandardNormalimain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2/shape"/device:GPU:0*
dtype0*
seed2 *

seed *
T0
?
gmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2/mulMulxmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2/RandomStandardNormaljmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2/stddev"/device:GPU:0*
T0
?
cmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2Addgmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2/mulhmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2/mean"/device:GPU:0*
T0
?
Ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/Abs_2Abscmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2"/device:GPU:0*
T0
?
Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/Sqrt_2SqrtYmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/Abs_2"/device:GPU:0*
T0
?
Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/Sign_2Signcmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/random_normal_2"/device:GPU:0*
T0
?
Ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/mul_2MulZmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/Sqrt_2Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/Sign_2"/device:GPU:0*
T0
?
Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/MatMulMatMulYmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/mul_1Ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/mul_2"/device:GPU:0*
transpose_a( *
transpose_b( *
T0
?
Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_2Muldmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/readWmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/mul"/device:GPU:0*
T0
?
Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_3Addbmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/readUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_2"/device:GPU:0*
T0
?
Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_3Mulfmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/readZmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/MatMul"/device:GPU:0*
T0
?
Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_4Adddmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/readUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_3"/device:GPU:0*
T0
?
Xmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1MatMulTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/ReluUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_4"/device:GPU:0*
transpose_a( *
transpose_b( *
T0
?
Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5AddXmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_3"/device:GPU:0*
T0
?
Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/ShapeShapeLmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice"/device:GPU:0*
T0*
out_type0
?
cmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/strided_slice/stackConst"/device:GPU:0*
valueB: *
dtype0
?
emain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/strided_slice/stack_1Const"/device:GPU:0*
valueB:*
dtype0
?
emain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/strided_slice/stack_2Const"/device:GPU:0*
valueB:*
dtype0
?
]main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/strided_sliceStridedSliceUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Shapecmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/strided_slice/stackemain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/strided_slice/stack_1emain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/strided_slice/stack_2"/device:GPU:0*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
?
_main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape/shape/1Const"/device:GPU:0*
dtype0*
value	B :
?
_main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape/shape/2Const"/device:GPU:0*
value	B :3*
dtype0
?
]main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape/shapePack]main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/strided_slice_main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape/shape/1_main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape/shape/2"/device:GPU:0*
T0*

axis *
N
?
Wmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/ReshapeReshapeUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5]main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape/shape"/device:GPU:0*
T0*
Tshape0
?
fmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean/reduction_indicesConst"/device:GPU:0*
value	B :*
dtype0
?
Tmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MeanMeanWmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshapefmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean/reduction_indices"/device:GPU:0*

Tidx0*
	keep_dims(*
T0
?
Smain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/subSubWmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/ReshapeTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean"/device:GPU:0*
T0
?
Emain_level/agent/main/target/network_0/rainbow_q_values_head_0/outputAddUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDimsSmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub"/device:GPU:0*
T0
?
Fmain_level/agent/main/target/network_0/rainbow_q_values_head_0/SoftmaxSoftmaxEmain_level/agent/main/target/network_0/rainbow_q_values_head_0/output"/device:GPU:0*
T0
?
Lmain_level/agent/main/target/network_0/rainbow_q_values_head_0/distributionsPlaceholder"/device:GPU:0* 
shape:?????????3*
dtype0
?
xmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradientLmain_level/agent/main/target/network_0/rainbow_q_values_head_0/distributions"/device:GPU:0*
T0
?
hmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/RankConst"/device:GPU:0*
value	B :*
dtype0
?
imain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/ShapeShapeEmain_level/agent/main/target/network_0/rainbow_q_values_head_0/output"/device:GPU:0*
T0*
out_type0
?
jmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Rank_1Const"/device:GPU:0*
dtype0*
value	B :
?
kmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Shape_1ShapeEmain_level/agent/main/target/network_0/rainbow_q_values_head_0/output"/device:GPU:0*
T0*
out_type0
?
imain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub/yConst"/device:GPU:0*
dtype0*
value	B :
?
gmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/SubSubjmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Rank_1imain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub/y"/device:GPU:0*
T0
?
omain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice/beginPackgmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub"/device:GPU:0*
T0*

axis *
N
?
nmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice/sizeConst"/device:GPU:0*
valueB:*
dtype0
?
imain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/SliceSlicekmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Shape_1omain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice/beginnmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice/size"/device:GPU:0*
T0*
Index0
?
smain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat/values_0Const"/device:GPU:0*
valueB:
?????????*
dtype0
?
omain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat/axisConst"/device:GPU:0*
value	B : *
dtype0
?
jmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concatConcatV2smain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat/values_0imain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sliceomain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat/axis"/device:GPU:0*

Tidx0*
T0*
N
?
kmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/ReshapeReshapeEmain_level/agent/main/target/network_0/rainbow_q_values_head_0/outputjmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat"/device:GPU:0*
T0*
Tshape0
?
jmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Rank_2Const"/device:GPU:0*
value	B :*
dtype0
?
kmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Shape_2Shapexmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/labels_stop_gradient"/device:GPU:0*
T0*
out_type0
?
kmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub_1/yConst"/device:GPU:0*
value	B :*
dtype0
?
imain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub_1Subjmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Rank_2kmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub_1/y"/device:GPU:0*
T0
?
qmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_1/beginPackimain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub_1"/device:GPU:0*
T0*

axis *
N
?
pmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst"/device:GPU:0*
valueB:*
dtype0
?
kmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_1Slicekmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Shape_2qmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_1/beginpmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_1/size"/device:GPU:0*
T0*
Index0
?
umain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat_1/values_0Const"/device:GPU:0*
valueB:
?????????*
dtype0
?
qmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat_1/axisConst"/device:GPU:0*
value	B : *
dtype0
?
lmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat_1ConcatV2umain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat_1/values_0kmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_1qmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat_1/axis"/device:GPU:0*
T0*
N*

Tidx0
?
mmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_1Reshapexmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/labels_stop_gradientlmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/concat_1"/device:GPU:0*
T0*
Tshape0
?
cmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogitskmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshapemmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_1"/device:GPU:0*
T0
?
kmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub_2/yConst"/device:GPU:0*
value	B :*
dtype0
?
imain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub_2Subhmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Rankkmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub_2/y"/device:GPU:0*
T0
?
qmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_2/beginConst"/device:GPU:0*
valueB: *
dtype0
?
pmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_2/sizePackimain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Sub_2"/device:GPU:0*
N*
T0*

axis 
?
kmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_2Sliceimain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Shapeqmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_2/beginpmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_2/size"/device:GPU:0*
T0*
Index0
?
mmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_2Reshapecmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sgkmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Slice_2"/device:GPU:0*
T0*
Tshape0
?
Cmain_level/agent/main/target/network_0/rainbow_q_values_head_0/CastCastFmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax"/device:GPU:0*
Truncate( *

DstT0*

SrcT0
?
Mmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/axesConst"/device:GPU:0*
valueB:*
dtype0
?
Mmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/freeConst"/device:GPU:0*
valueB"       *
dtype0
?
Nmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/ShapeShapeCmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Cast"/device:GPU:0*
T0*
out_type0
?
Vmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/GatherV2/axisConst"/device:GPU:0*
value	B : *
dtype0
?
Qmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/GatherV2GatherV2Nmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/ShapeMmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/freeVmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/GatherV2/axis"/device:GPU:0*
Tindices0*
Tparams0*
Taxis0
?
Xmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/GatherV2_1/axisConst"/device:GPU:0*
dtype0*
value	B : 
?
Smain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/GatherV2_1GatherV2Nmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/ShapeMmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/axesXmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/GatherV2_1/axis"/device:GPU:0*
Tparams0*
Taxis0*
Tindices0
?
Nmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/ConstConst"/device:GPU:0*
valueB: *
dtype0
?
Mmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/ProdProdQmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/GatherV2Nmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
Pmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/Const_1Const"/device:GPU:0*
valueB: *
dtype0
?
Omain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/Prod_1ProdSmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/GatherV2_1Pmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/Const_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
Tmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/concat/axisConst"/device:GPU:0*
value	B : *
dtype0
?
Omain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/concatConcatV2Mmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/freeMmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/axesTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/concat/axis"/device:GPU:0*
T0*
N*

Tidx0
?
Nmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/stackPackMmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/ProdOmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/Prod_1"/device:GPU:0*
N*
T0*

axis 
?
Rmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/transpose	TransposeCmain_level/agent/main/target/network_0/rainbow_q_values_head_0/CastOmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/concat"/device:GPU:0*
T0*
Tperm0
?
Pmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/ReshapeReshapeRmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/transposeNmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/stack"/device:GPU:0*
T0*
Tshape0
?
Ymain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/transpose_1/permConst"/device:GPU:0*
valueB: *
dtype0
?
Tmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/transpose_1	Transpose+main_level/agent/main/target/network_0/CastYmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/transpose_1/perm"/device:GPU:0*
Tperm0*
T0
?
Xmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/Reshape_1/shapeConst"/device:GPU:0*
valueB"3      *
dtype0
?
Rmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/Reshape_1ReshapeTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/transpose_1Xmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/Reshape_1/shape"/device:GPU:0*
T0*
Tshape0
?
Omain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/MatMulMatMulPmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/ReshapeRmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/Reshape_1"/device:GPU:0*
transpose_a( *
transpose_b( *
T0
?
Pmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/Const_2Const"/device:GPU:0*
valueB *
dtype0
?
Vmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/concat_1/axisConst"/device:GPU:0*
value	B : *
dtype0
?
Qmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/concat_1ConcatV2Qmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/GatherV2Pmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/Const_2Vmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/concat_1/axis"/device:GPU:0*

Tidx0*
T0*
N
?
Hmain_level/agent/main/target/network_0/rainbow_q_values_head_0/TensordotReshapeOmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/MatMulQmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Tensordot/concat_1"/device:GPU:0*
T0*
Tshape0
?
Hmain_level/agent/main/target/network_0/rainbow_q_values_head_0/truediv/yConst"/device:GPU:0*
valueB 2      ??*
dtype0
?
Fmain_level/agent/main/target/network_0/rainbow_q_values_head_0/truedivRealDivHmain_level/agent/main/target/network_0/rainbow_q_values_head_0/TensordotHmain_level/agent/main/target/network_0/rainbow_q_values_head_0/truediv/y"/device:GPU:0*
T0
?
Hmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_1SoftmaxFmain_level/agent/main/target/network_0/rainbow_q_values_head_0/truediv"/device:GPU:0*
T0
?
hmain_level/agent/main/target/network_0/rainbow_q_values_head_0/rainbow_q_values_head_0_importance_weightPlaceholder"/device:GPU:0*
dtype0* 
shape:?????????
?
(main_level/agent/main/target/Rank/packedPackmmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_2"/device:GPU:0*
T0*

axis *
N
Z
!main_level/agent/main/target/RankConst"/device:GPU:0*
value	B :*
dtype0
a
(main_level/agent/main/target/range/startConst"/device:GPU:0*
value	B : *
dtype0
a
(main_level/agent/main/target/range/deltaConst"/device:GPU:0*
value	B :*
dtype0
?
"main_level/agent/main/target/rangeRange(main_level/agent/main/target/range/start!main_level/agent/main/target/Rank(main_level/agent/main/target/range/delta"/device:GPU:0*

Tidx0
?
&main_level/agent/main/target/Sum/inputPackmmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_2"/device:GPU:0*
N*
T0*

axis 
?
 main_level/agent/main/target/SumSum&main_level/agent/main/target/Sum/input"main_level/agent/main/target/range"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
g
%main_level/agent/main/target/0_holderPlaceholder"/device:GPU:0*
dtype0*
shape:
??
b
%main_level/agent/main/target/1_holderPlaceholder"/device:GPU:0*
shape:?*
dtype0
g
%main_level/agent/main/target/2_holderPlaceholder"/device:GPU:0*
shape:
??*
dtype0
b
%main_level/agent/main/target/3_holderPlaceholder"/device:GPU:0*
dtype0*
shape:?
]
%main_level/agent/main/target/4_holderPlaceholder"/device:GPU:0*
shape: *
dtype0
g
%main_level/agent/main/target/5_holderPlaceholder"/device:GPU:0*
shape:
??*
dtype0
b
%main_level/agent/main/target/6_holderPlaceholder"/device:GPU:0*
dtype0*
shape:?
g
%main_level/agent/main/target/7_holderPlaceholder"/device:GPU:0*
dtype0*
shape:
??
b
%main_level/agent/main/target/8_holderPlaceholder"/device:GPU:0*
shape:?*
dtype0
f
%main_level/agent/main/target/9_holderPlaceholder"/device:GPU:0*
dtype0*
shape:	?3
b
&main_level/agent/main/target/10_holderPlaceholder"/device:GPU:0*
dtype0*
shape:3
g
&main_level/agent/main/target/11_holderPlaceholder"/device:GPU:0*
dtype0*
shape:	?3
b
&main_level/agent/main/target/12_holderPlaceholder"/device:GPU:0*
dtype0*
shape:3
h
&main_level/agent/main/target/13_holderPlaceholder"/device:GPU:0*
dtype0*
shape:
??
c
&main_level/agent/main/target/14_holderPlaceholder"/device:GPU:0*
dtype0*
shape:?
h
&main_level/agent/main/target/15_holderPlaceholder"/device:GPU:0*
dtype0*
shape:
??
c
&main_level/agent/main/target/16_holderPlaceholder"/device:GPU:0*
shape:?*
dtype0
h
&main_level/agent/main/target/17_holderPlaceholder"/device:GPU:0*
dtype0*
shape:
??
c
&main_level/agent/main/target/18_holderPlaceholder"/device:GPU:0*
shape:?*
dtype0
h
&main_level/agent/main/target/19_holderPlaceholder"/device:GPU:0*
shape:
??*
dtype0
c
&main_level/agent/main/target/20_holderPlaceholder"/device:GPU:0*
shape:?*
dtype0
?
%main_level/agent/main/target/Assign_1AssignNmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean%main_level/agent/main/target/0_holder"/device:GPU:0*
use_locking( *
T0*a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean*
validate_shape(
?
%main_level/agent/main/target/Assign_2AssignLmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_mean%main_level/agent/main/target/1_holder"/device:GPU:0*
use_locking( *
T0*_
_classU
SQloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_mean*
validate_shape(
?
%main_level/agent/main/target/Assign_3AssignPmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev%main_level/agent/main/target/2_holder"/device:GPU:0*
use_locking( *
T0*c
_classY
WUloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev*
validate_shape(
?
%main_level/agent/main/target/Assign_4AssignNmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev%main_level/agent/main/target/3_holder"/device:GPU:0*
use_locking( *
T0*a
_classW
USloc:@main_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev*
validate_shape(
?
%main_level/agent/main/target/Assign_5AssignHmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers%main_level/agent/main/target/4_holder"/device:GPU:0*
use_locking( *
T0*[
_classQ
OMloc:@main_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers*
validate_shape(
?
%main_level/agent/main/target/Assign_6AssignZmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean%main_level/agent/main/target/5_holder"/device:GPU:0*
use_locking( *
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean*
validate_shape(
?
%main_level/agent/main/target/Assign_7AssignXmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean%main_level/agent/main/target/6_holder"/device:GPU:0*
validate_shape(*
use_locking( *
T0*k
_classa
_]loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean
?
%main_level/agent/main/target/Assign_8Assign\main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev%main_level/agent/main/target/7_holder"/device:GPU:0*o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev*
validate_shape(*
use_locking( *
T0
?
%main_level/agent/main/target/Assign_9AssignZmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev%main_level/agent/main/target/8_holder"/device:GPU:0*
use_locking( *
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev*
validate_shape(
?
&main_level/agent/main/target/Assign_10AssignZmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean%main_level/agent/main/target/9_holder"/device:GPU:0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean*
validate_shape(*
use_locking( *
T0
?
&main_level/agent/main/target/Assign_11AssignXmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean&main_level/agent/main/target/10_holder"/device:GPU:0*
use_locking( *
T0*k
_classa
_]loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean*
validate_shape(
?
&main_level/agent/main/target/Assign_12Assign\main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev&main_level/agent/main/target/11_holder"/device:GPU:0*
use_locking( *
T0*o
_classe
caloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev*
validate_shape(
?
&main_level/agent/main/target/Assign_13AssignZmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev&main_level/agent/main/target/12_holder"/device:GPU:0*
use_locking( *
T0*m
_classc
a_loc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev*
validate_shape(
?
&main_level/agent/main/target/Assign_14Assign_main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean&main_level/agent/main/target/13_holder"/device:GPU:0*
use_locking( *
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean*
validate_shape(
?
&main_level/agent/main/target/Assign_15Assign]main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean&main_level/agent/main/target/14_holder"/device:GPU:0*
use_locking( *
T0*p
_classf
dbloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean*
validate_shape(
?
&main_level/agent/main/target/Assign_16Assignamain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev&main_level/agent/main/target/15_holder"/device:GPU:0*
T0*t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev*
validate_shape(*
use_locking( 
?
&main_level/agent/main/target/Assign_17Assign_main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev&main_level/agent/main/target/16_holder"/device:GPU:0*
use_locking( *
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev*
validate_shape(
?
&main_level/agent/main/target/Assign_18Assign_main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean&main_level/agent/main/target/17_holder"/device:GPU:0*
use_locking( *
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean*
validate_shape(
?
&main_level/agent/main/target/Assign_19Assign]main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean&main_level/agent/main/target/18_holder"/device:GPU:0*
use_locking( *
T0*p
_classf
dbloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean*
validate_shape(
?
&main_level/agent/main/target/Assign_20Assignamain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev&main_level/agent/main/target/19_holder"/device:GPU:0*
use_locking( *
T0*t
_classj
hfloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev*
validate_shape(
?
&main_level/agent/main/target/Assign_21Assign_main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev&main_level/agent/main/target/20_holder"/device:GPU:0*
use_locking( *
T0*r
_classh
fdloc:@main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev*
validate_shape(
d
,main_level/agent/main/target/gradients/ShapeConst"/device:GPU:0*
valueB *
dtype0
l
0main_level/agent/main/target/gradients/grad_ys_0Const"/device:GPU:0*
valueB
 *  ??*
dtype0
?
+main_level/agent/main/target/gradients/FillFill,main_level/agent/main/target/gradients/Shape0main_level/agent/main/target/gradients/grad_ys_0"/device:GPU:0*
T0*

index_type0
?
Zmain_level/agent/main/target/gradients/main_level/agent/main/target/Sum_grad/Reshape/shapeConst"/device:GPU:0*!
valueB"         *
dtype0
?
Tmain_level/agent/main/target/gradients/main_level/agent/main/target/Sum_grad/ReshapeReshape+main_level/agent/main/target/gradients/FillZmain_level/agent/main/target/gradients/main_level/agent/main/target/Sum_grad/Reshape/shape"/device:GPU:0*
T0*
Tshape0
?
Rmain_level/agent/main/target/gradients/main_level/agent/main/target/Sum_grad/ShapeShape&main_level/agent/main/target/Sum/input"/device:GPU:0*
T0*
out_type0
?
Qmain_level/agent/main/target/gradients/main_level/agent/main/target/Sum_grad/TileTileTmain_level/agent/main/target/gradients/main_level/agent/main/target/Sum_grad/ReshapeRmain_level/agent/main/target/gradients/main_level/agent/main/target/Sum_grad/Shape"/device:GPU:0*

Tmultiples0*
T0
?
Zmain_level/agent/main/target/gradients/main_level/agent/main/target/Sum/input_grad/unstackUnpackQmain_level/agent/main/target/gradients/main_level/agent/main/target/Sum_grad/Tile"/device:GPU:0*
T0*	
num*

axis 
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShapecmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshapeZmain_level/agent/main/target/gradients/main_level/agent/main/target/Sum/input_grad/unstack?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
1main_level/agent/main/target/gradients/zeros_like	ZerosLikeemain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg:1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst"/device:GPU:0*
valueB :
?????????*
dtype0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDims?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Reshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim"/device:GPU:0*
T0*

Tdim0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/mulMul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/ExpandDimsemain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg:1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmaxkmain_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/NegNeg?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst"/device:GPU:0*
valueB :
?????????*
dtype0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDims?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Reshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim"/device:GPU:0*

Tdim0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/mul_1Mul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/Neg"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapeEmain_level/agent/main/target/network_0/rainbow_q_values_head_0/output"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg_grad/mul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
wmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/ShapeShapeUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDims"/device:GPU:0*
T0*
out_type0
?
ymain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Shape_1ShapeSmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/BroadcastGradientArgsBroadcastGradientArgswmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Shapeymain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Shape_1"/device:GPU:0*
T0
?
umain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/SumSum?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
ymain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/ReshapeReshapeumain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Sumwmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
wmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Sum_1Sum?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
{main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Reshape_1Reshapewmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Sum_1ymain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/ShapeShapePmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/ReshapeReshapeymain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Reshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/ShapeShapeWmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape_1ShapeTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/SumSum{main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Reshape_1?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/ReshapeReshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Sum?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Sum_1Sum{main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Reshape_1?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/NegNeg?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Sum_1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape_1Reshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Neg?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/ShapeShapeSmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_1"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape_1Const"/device:GPU:0*
valueB:3*
dtype0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/SumSum?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/Reshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/ReshapeReshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Sum?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Sum_1Sum?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/Reshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape_1Reshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Sum_1?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape_1"/device:GPU:0*
Tshape0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ShapeShapeWmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/SizeConst"/device:GPU:0*
dtype0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/addAddfmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean/reduction_indices?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Size"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/modFloorMod?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/add?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Size"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_1Const"/device:GPU:0*
dtype0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
valueB 
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/startConst"/device:GPU:0*
dtype0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B : 
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/deltaConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :*
dtype0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/rangeRange?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/start?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Size?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/delta"/device:GPU:0*

Tidx0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Fill/valueConst"/device:GPU:0*
dtype0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/FillFill?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_1?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Fill/value"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*

index_type0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/DynamicStitchDynamicStitch?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/mod?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Fill"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
N
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum/yConst"/device:GPU:0*
dtype0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/MaximumMaximum?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/DynamicStitch?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum/y"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordivFloorDiv?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ReshapeReshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape_1?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/DynamicStitch"/device:GPU:0*
Tshape0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/TileTile?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Reshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordiv"/device:GPU:0*

Tmultiples0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_2ShapeWmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_3ShapeTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ConstConst"/device:GPU:0*
valueB: *
dtype0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ProdProd?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_2?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Const"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Const_1Const"/device:GPU:0*
dtype0*
valueB: 
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Prod_1Prod?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_3?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Const_1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1/yConst"/device:GPU:0*
value	B :*
dtype0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1Maximum?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Prod_1?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1/y"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordiv_1FloorDiv?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Prod?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/CastCast?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordiv_1"/device:GPU:0*

SrcT0*
Truncate( *

DstT0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/truedivRealDiv?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Tile?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Cast"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMulMatMul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/ReshapePmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_4"/device:GPU:0*
transpose_a( *
transpose_b(*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMul_1MatMulOmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/Relu?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape"/device:GPU:0*
transpose_a(*
transpose_b( *
T0
?
+main_level/agent/main/target/gradients/AddNAddN?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/truediv"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape*
N
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/ShapeShapeUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/ReshapeReshape+main_level/agent/main/target/gradients/AddN?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/Relu_grad/ReluGradReluGrad?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMulOmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/Relu"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_2_grad/MulMul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape_1Rmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/mul"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_2_grad/Mul_1Mul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape_1_main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/ShapeShapeXmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape_1Const"/device:GPU:0*
valueB:?*
dtype0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/SumSum?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/Reshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/ReshapeReshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Sum?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Sum_1Sum?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/Reshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape_1Reshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Sum_1?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/ShapeShapeQmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul"/device:GPU:0*
out_type0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape_1Const"/device:GPU:0*
valueB:?*
dtype0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/SumSum?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/Relu_grad/ReluGrad?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/ReshapeReshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Sum?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Sum_1Sum?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/Relu_grad/ReluGrad?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape_1Reshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Sum_1?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_3_grad/MulMul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMul_1Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/MatMul"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_3_grad/Mul_1Mul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMul_1amain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMulMatMul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/ReshapeUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_4"/device:GPU:0*
transpose_a( *
transpose_b(*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMul_1MatMulTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Relu?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMulMatMul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/ReshapePmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_1"/device:GPU:0*
T0*
transpose_a( *
transpose_b(
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul_1MatMulLmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape"/device:GPU:0*
T0*
transpose_a(*
transpose_b( 
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Relu_grad/ReluGradReluGrad?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMulTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Relu"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_2_grad/MulMul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape_1Wmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/mul"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_2_grad/Mul_1Mul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape_1dmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/read"/device:GPU:0*
T0
?
~main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_grad/MulMul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape_1Rmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/mul"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_grad/Mul_1Mul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape_1_main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/ShapeShapeVmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape_1Const"/device:GPU:0*
valueB:?*
dtype0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/SumSum?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Relu_grad/ReluGrad?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/ReshapeReshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Sum?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Sum_1Sum?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Relu_grad/ReluGrad?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape_1Reshape?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Sum_1?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_3_grad/MulMul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMul_1Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/MatMul"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_3_grad/Mul_1Mul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMul_1fmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_1_grad/MulMul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul_1Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/MatMul"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_1_grad/Mul_1Mul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul_1amain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMulMatMul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/ReshapeUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_1"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul_1MatMulLmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape"/device:GPU:0*
T0*
transpose_a(*
transpose_b( 
?
-main_level/agent/main/target/gradients/AddN_1AddN?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul*
N
?
~main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice_grad/ShapeShape*main_level/agent/main/target/network_0/add"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice_grad/StridedSliceGradStridedSliceGrad~main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice_grad/ShapeRmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice/stackTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice/stack_1Tmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice/stack_2-main_level/agent/main/target/gradients/AddN_1"/device:GPU:0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_grad/MulMul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape_1Wmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/mul"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_grad/Mul_1Mul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape_1dmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/read"/device:GPU:0*
T0
?
\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/ShapeShape*main_level/agent/main/target/network_0/mul"/device:GPU:0*
T0*
out_type0
?
^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Shape_1Shape,main_level/agent/main/target/network_0/mul_1"/device:GPU:0*
T0*
out_type0
?
lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/BroadcastGradientArgsBroadcastGradientArgs\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Shape^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Shape_1"/device:GPU:0*
T0
?
Zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/SumSum?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice_grad/StridedSliceGradlmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/ReshapeReshapeZmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Sum\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Shape"/device:GPU:0*
Tshape0*
T0
?
\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Sum_1Sum?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice_grad/StridedSliceGradnmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
`main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Reshape_1Reshape\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Sum_1^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Shape_1"/device:GPU:0*
Tshape0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_1_grad/MulMul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul_1Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/MatMul"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_1_grad/Mul_1Mul?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul_1fmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/read"/device:GPU:0*
T0
?
\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
?
^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Shape_1Shape3main_level/agent/main/target/network_0/StopGradient"/device:GPU:0*
T0*
out_type0
?
lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/BroadcastGradientArgsBroadcastGradientArgs\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Shape^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Shape_1"/device:GPU:0*
T0
?
Zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/MulMul^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Reshape3main_level/agent/main/target/network_0/StopGradient"/device:GPU:0*
T0
?
Zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/SumSumZmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Mullmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/BroadcastGradientArgs"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/ReshapeReshapeZmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Sum\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Mul_1Mul*main_level/agent/main/target/network_0/sub^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Reshape"/device:GPU:0*
T0
?
\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Sum_1Sum\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Mul_1nmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
`main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Reshape_1Reshape\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Sum_1^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
?
`main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Shape_1Shape.main_level/agent/main/target/network_0/mul_1/y"/device:GPU:0*
T0*
out_type0
?
nmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Shape`main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0
?
\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/MulMul`main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Reshape_1.main_level/agent/main/target/network_0/mul_1/y"/device:GPU:0*
T0
?
\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/SumSum\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Mulnmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
`main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/ReshapeReshape\main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Sum^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Mul_1MulMmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers/read`main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/add_grad/Reshape_1"/device:GPU:0*
T0
?
^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Sum_1Sum^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Mul_1pmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
bmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Reshape_1Reshape^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Sum_1`main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
Zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/sub_grad/NegNeg^main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_grad/Reshape"/device:GPU:0*
T0
?
bmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1/y_grad/unstackUnpackbmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Reshape_1"/device:GPU:0*
T0*	
num*

axis 
?
-main_level/agent/main/target/gradients/AddN_2AddN`main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/ReshapeZmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/sub_grad/Neg"/device:GPU:0*
T0*s
_classi
geloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Reshape*
N
?
|main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/ShapeShapeZmain_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0
?
~main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/ReshapeReshapebmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1/y_grad/unstack|main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation_grad/ReluGradReluGrad~main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/ReshapeZmain_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0
?
jmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/add_2_grad/ShapeShape9main_level/agent/main/target/network_0/observation/MatMul"/device:GPU:0*
T0*
out_type0
?
lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/add_2_grad/Shape_1Const"/device:GPU:0*
valueB:?*
dtype0
?
zmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsjmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/add_2_grad/Shapelmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/add_2_grad/Shape_1"/device:GPU:0*
T0
?
hmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/add_2_grad/SumSum?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation_grad/ReluGradzmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/add_2_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/add_2_grad/ReshapeReshapehmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/add_2_grad/Sumjmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/add_2_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
jmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/add_2_grad/Sum_1Sum?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation_grad/ReluGrad|main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/add_2_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
nmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/add_2_grad/Reshape_1Reshapejmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/add_2_grad/Sum_1lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/add_2_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
lmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/MatMul_grad/MatMulMatMullmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/add_2_grad/Reshape8main_level/agent/main/target/network_0/observation/add_1"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
?
nmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/MatMul_grad/MatMul_1MatMul6main_level/agent/main/target/network_0/observation/sublmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/add_2_grad/Reshape"/device:GPU:0*
T0*
transpose_a(*
transpose_b( 
?
fmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/mul_grad/MulMulnmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/add_2_grad/Reshape_1Fmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/mul"/device:GPU:0*
T0
?
hmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/mul_grad/Mul_1Mulnmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/add_2_grad/Reshape_1Smain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev/read"/device:GPU:0*
T0
?
hmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/mul_1_grad/MulMulnmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/MatMul_grad/MatMul_1Imain_level/agent/main/target/network_0/observation/NoisyNetDense_0/MatMul"/device:GPU:0*
T0
?
jmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/mul_1_grad/Mul_1Mulnmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/MatMul_grad/MatMul_1Umain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev/read"/device:GPU:0*
T0
?
/main_level/agent/main/target/global_norm/L2LossL2Lossnmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/MatMul_grad/MatMul_1"/device:GPU:0*
T0*?
_classw
usloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/MatMul_grad/MatMul_1
?
1main_level/agent/main/target/global_norm/L2Loss_1L2Lossnmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/add_2_grad/Reshape_1"/device:GPU:0*?
_classw
usloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/add_2_grad/Reshape_1*
T0
?
1main_level/agent/main/target/global_norm/L2Loss_2L2Losshmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/mul_1_grad/Mul"/device:GPU:0*
T0*{
_classq
omloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/mul_1_grad/Mul
?
1main_level/agent/main/target/global_norm/L2Loss_3L2Lossfmain_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/mul_grad/Mul"/device:GPU:0*
T0*y
_classo
mkloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/observation/mul_grad/Mul
?
1main_level/agent/main/target/global_norm/L2Loss_4L2Loss-main_level/agent/main/target/gradients/AddN_2"/device:GPU:0*
T0*s
_classi
geloc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/mul_1_grad/Reshape
?
1main_level/agent/main/target/global_norm/L2Loss_5L2Loss?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul_1"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul_1
?
1main_level/agent/main/target/global_norm/L2Loss_6L2Loss?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape_1"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape_1
?
1main_level/agent/main/target/global_norm/L2Loss_7L2Loss?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_1_grad/Mul"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_1_grad/Mul*
T0
?
1main_level/agent/main/target/global_norm/L2Loss_8L2Loss~main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_grad/Mul"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_grad/Mul*
T0
?
1main_level/agent/main/target/global_norm/L2Loss_9L2Loss?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMul_1"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMul_1
?
2main_level/agent/main/target/global_norm/L2Loss_10L2Loss?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape_1"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape_1
?
2main_level/agent/main/target/global_norm/L2Loss_11L2Loss?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_3_grad/Mul"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_3_grad/Mul
?
2main_level/agent/main/target/global_norm/L2Loss_12L2Loss?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_2_grad/Mul"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_2_grad/Mul
?
2main_level/agent/main/target/global_norm/L2Loss_13L2Loss?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul_1"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul_1
?
2main_level/agent/main/target/global_norm/L2Loss_14L2Loss?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape_1"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape_1
?
2main_level/agent/main/target/global_norm/L2Loss_15L2Loss?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_1_grad/Mul"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_1_grad/Mul
?
2main_level/agent/main/target/global_norm/L2Loss_16L2Loss?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_grad/Mul"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_grad/Mul
?
2main_level/agent/main/target/global_norm/L2Loss_17L2Loss?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMul_1"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMul_1
?
2main_level/agent/main/target/global_norm/L2Loss_18L2Loss?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape_1"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape_1
?
2main_level/agent/main/target/global_norm/L2Loss_19L2Loss?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_3_grad/Mul"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_3_grad/Mul
?
2main_level/agent/main/target/global_norm/L2Loss_20L2Loss?main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_2_grad/Mul"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/target/gradients/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_2_grad/Mul*
T0
?	
.main_level/agent/main/target/global_norm/stackPack/main_level/agent/main/target/global_norm/L2Loss1main_level/agent/main/target/global_norm/L2Loss_11main_level/agent/main/target/global_norm/L2Loss_21main_level/agent/main/target/global_norm/L2Loss_31main_level/agent/main/target/global_norm/L2Loss_41main_level/agent/main/target/global_norm/L2Loss_51main_level/agent/main/target/global_norm/L2Loss_61main_level/agent/main/target/global_norm/L2Loss_71main_level/agent/main/target/global_norm/L2Loss_81main_level/agent/main/target/global_norm/L2Loss_92main_level/agent/main/target/global_norm/L2Loss_102main_level/agent/main/target/global_norm/L2Loss_112main_level/agent/main/target/global_norm/L2Loss_122main_level/agent/main/target/global_norm/L2Loss_132main_level/agent/main/target/global_norm/L2Loss_142main_level/agent/main/target/global_norm/L2Loss_152main_level/agent/main/target/global_norm/L2Loss_162main_level/agent/main/target/global_norm/L2Loss_172main_level/agent/main/target/global_norm/L2Loss_182main_level/agent/main/target/global_norm/L2Loss_192main_level/agent/main/target/global_norm/L2Loss_20"/device:GPU:0*
T0*

axis *
N
k
.main_level/agent/main/target/global_norm/ConstConst"/device:GPU:0*
valueB: *
dtype0
?
,main_level/agent/main/target/global_norm/SumSum.main_level/agent/main/target/global_norm/stack.main_level/agent/main/target/global_norm/Const"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
l
0main_level/agent/main/target/global_norm/Const_1Const"/device:GPU:0*
valueB
 *   @*
dtype0
?
,main_level/agent/main/target/global_norm/mulMul,main_level/agent/main/target/global_norm/Sum0main_level/agent/main/target/global_norm/Const_1"/device:GPU:0*
T0
?
4main_level/agent/main/target/global_norm/global_normSqrt,main_level/agent/main/target/global_norm/mul"/device:GPU:0*
T0
?
.main_level/agent/main/target/gradients_1/ShapeShapeFmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax"/device:GPU:0*
T0*
out_type0
n
2main_level/agent/main/target/gradients_1/grad_ys_0Const"/device:GPU:0*
valueB
 *  ??*
dtype0
?
-main_level/agent/main/target/gradients_1/FillFill.main_level/agent/main/target/gradients_1/Shape2main_level/agent/main/target/gradients_1/grad_ys_0"/device:GPU:0*
T0*

index_type0
?
xmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/mulMul-main_level/agent/main/target/gradients_1/FillFmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/Sum/reduction_indicesConst"/device:GPU:0*
valueB :
?????????*
dtype0
?
xmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/SumSumxmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/mul?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/Sum/reduction_indices"/device:GPU:0*
T0*

Tidx0*
	keep_dims(
?
xmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/subSub-main_level/agent/main/target/gradients_1/Fillxmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/Sum"/device:GPU:0*
T0
?
zmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/mul_1Mulxmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/subFmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax"/device:GPU:0*
T0
?
ymain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/ShapeShapeUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDims"/device:GPU:0*
T0*
out_type0
?
{main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Shape_1ShapeSmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/BroadcastGradientArgsBroadcastGradientArgsymain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Shape{main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Shape_1"/device:GPU:0*
T0
?
wmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/SumSumzmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/mul_1?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/BroadcastGradientArgs"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
{main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/ReshapeReshapewmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Sumymain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
ymain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Sum_1Sumzmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/mul_1?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
}main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Reshape_1Reshapeymain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Sum_1{main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/ShapeShapePmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/ReshapeReshape{main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Reshape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/ShapeShapeWmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape_1ShapeTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/SumSum}main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Reshape_1?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/ReshapeReshape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Sum?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Sum_1Sum}main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Reshape_1?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/NegNeg?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Sum_1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape_1Reshape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Neg?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/ShapeShapeSmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_1"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape_1Const"/device:GPU:0*
valueB:3*
dtype0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/SumSum?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/Reshape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/ReshapeReshape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Sum?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Sum_1Sum?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/Reshape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape_1Reshape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Sum_1?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ShapeShapeWmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape"/device:GPU:0*
out_type0*
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/SizeConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :*
dtype0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/addAddfmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean/reduction_indices?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Size"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/modFloorMod?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/add?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Size"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_1Const"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
valueB *
dtype0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/startConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B : *
dtype0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/deltaConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :*
dtype0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/rangeRange?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/start?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Size?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/delta"/device:GPU:0*

Tidx0*?
_class?
??loc:@main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Fill/valueConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :*
dtype0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/FillFill?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_1?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Fill/value"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*

index_type0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/DynamicStitchDynamicStitch?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/mod?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Fill"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
N
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum/yConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :*
dtype0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/MaximumMaximum?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/DynamicStitch?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum/y"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordivFloorDiv?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ReshapeReshape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape_1?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/DynamicStitch"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/TileTile?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Reshape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordiv"/device:GPU:0*

Tmultiples0*
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_2ShapeWmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape"/device:GPU:0*
out_type0*
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_3ShapeTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean"/device:GPU:0*
out_type0*
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ConstConst"/device:GPU:0*
valueB: *
dtype0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ProdProd?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_2?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Const_1Const"/device:GPU:0*
valueB: *
dtype0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Prod_1Prod?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_3?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Const_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1/yConst"/device:GPU:0*
value	B :*
dtype0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1Maximum?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Prod_1?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1/y"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordiv_1FloorDiv?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Prod?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/CastCast?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordiv_1"/device:GPU:0*
Truncate( *

DstT0*

SrcT0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/truedivRealDiv?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Tile?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Cast"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMulMatMul?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/ReshapePmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_4"/device:GPU:0*
transpose_a( *
transpose_b(*
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMul_1MatMulOmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/Relu?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
?
-main_level/agent/main/target/gradients_1/AddNAddN?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/truediv"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape*
N
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/ShapeShapeUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/ReshapeReshape-main_level/agent/main/target/gradients_1/AddN?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/Shape"/device:GPU:0*
Tshape0*
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/Relu_grad/ReluGradReluGrad?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMulOmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/Relu"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/ShapeShapeXmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape_1Const"/device:GPU:0*
valueB:?*
dtype0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/SumSum?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/Reshape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/BroadcastGradientArgs"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/ReshapeReshape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Sum?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Sum_1Sum?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/Reshape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape_1Reshape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Sum_1?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape_1"/device:GPU:0*
Tshape0*
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/ShapeShapeQmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape_1Const"/device:GPU:0*
valueB:?*
dtype0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/SumSum?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/Relu_grad/ReluGrad?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/ReshapeReshape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Sum?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Sum_1Sum?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/Relu_grad/ReluGrad?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape_1Reshape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Sum_1?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMulMatMul?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/ReshapeUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_4"/device:GPU:0*
T0*
transpose_a( *
transpose_b(
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMul_1MatMulTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Relu?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape"/device:GPU:0*
T0*
transpose_a(*
transpose_b( 
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMulMatMul?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/ReshapePmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_1"/device:GPU:0*
T0*
transpose_a( *
transpose_b(
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul_1MatMulLmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape"/device:GPU:0*
T0*
transpose_a(*
transpose_b( 
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Relu_grad/ReluGradReluGrad?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMulTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Relu"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/ShapeShapeVmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape_1Const"/device:GPU:0*
valueB:?*
dtype0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/SumSum?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Relu_grad/ReluGrad?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/ReshapeReshape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Sum?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Sum_1Sum?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Relu_grad/ReluGrad?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape_1Reshape?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Sum_1?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMulMatMul?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/ReshapeUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_1"/device:GPU:0*
transpose_a( *
transpose_b(*
T0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul_1MatMulLmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape"/device:GPU:0*
T0*
transpose_a(*
transpose_b( 
?
/main_level/agent/main/target/gradients_1/AddN_1AddN?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul*
N
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice_grad/ShapeShape*main_level/agent/main/target/network_0/add"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice_grad/StridedSliceGradStridedSliceGrad?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice_grad/ShapeRmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice/stackTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice/stack_1Tmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice/stack_2/main_level/agent/main/target/gradients_1/AddN_1"/device:GPU:0*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
?
^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/ShapeShape*main_level/agent/main/target/network_0/mul"/device:GPU:0*
T0*
out_type0
?
`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Shape_1Shape,main_level/agent/main/target/network_0/mul_1"/device:GPU:0*
T0*
out_type0
?
nmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Shape`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Shape_1"/device:GPU:0*
T0
?
\main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/SumSum?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice_grad/StridedSliceGradnmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/ReshapeReshape\main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Sum^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Sum_1Sum?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice_grad/StridedSliceGradpmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
bmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Reshape_1Reshape^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Sum_1`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
?
`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Shape_1Shape3main_level/agent/main/target/network_0/StopGradient"/device:GPU:0*
T0*
out_type0
?
nmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Shape`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Shape_1"/device:GPU:0*
T0
?
\main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/MulMul`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Reshape3main_level/agent/main/target/network_0/StopGradient"/device:GPU:0*
T0
?
\main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/SumSum\main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Mulnmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/ReshapeReshape\main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Sum^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Mul_1Mul*main_level/agent/main/target/network_0/sub`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Reshape"/device:GPU:0*
T0
?
^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Sum_1Sum^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Mul_1pmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
bmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Reshape_1Reshape^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Sum_1`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
?
bmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Shape_1Shape.main_level/agent/main/target/network_0/mul_1/y"/device:GPU:0*
out_type0*
T0
?
pmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Shapebmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0
?
^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/MulMulbmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Reshape_1.main_level/agent/main/target/network_0/mul_1/y"/device:GPU:0*
T0
?
^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/SumSum^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Mulpmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
bmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/ReshapeReshape^main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Sum`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Shape"/device:GPU:0*
Tshape0*
T0
?
`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Mul_1MulMmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers/readbmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/add_grad/Reshape_1"/device:GPU:0*
T0
?
`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Sum_1Sum`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Mul_1rmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
dmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Reshape_1Reshape`main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Sum_1bmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Shape_1"/device:GPU:0*
Tshape0*
T0
?
dmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1/y_grad/unstackUnpackdmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1_grad/Reshape_1"/device:GPU:0*
T0*	
num*

axis 
?
~main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/ShapeShapeZmain_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/ReshapeReshapedmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/mul_1/y_grad/unstack~main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation_grad/ReluGradReluGrad?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/ReshapeZmain_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0
?
lmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/add_2_grad/ShapeShape9main_level/agent/main/target/network_0/observation/MatMul"/device:GPU:0*
T0*
out_type0
?
nmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/add_2_grad/Shape_1Const"/device:GPU:0*
valueB:?*
dtype0
?
|main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/add_2_grad/BroadcastGradientArgsBroadcastGradientArgslmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/add_2_grad/Shapenmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/add_2_grad/Shape_1"/device:GPU:0*
T0
?
jmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/add_2_grad/SumSum?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation_grad/ReluGrad|main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/add_2_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
nmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/add_2_grad/ReshapeReshapejmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/add_2_grad/Sumlmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/add_2_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
lmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/add_2_grad/Sum_1Sum?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation_grad/ReluGrad~main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/add_2_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
pmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/add_2_grad/Reshape_1Reshapelmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/add_2_grad/Sum_1nmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/add_2_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
nmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/MatMul_grad/MatMulMatMulnmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/add_2_grad/Reshape8main_level/agent/main/target/network_0/observation/add_1"/device:GPU:0*
T0*
transpose_a( *
transpose_b(
?
pmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/MatMul_grad/MatMul_1MatMul6main_level/agent/main/target/network_0/observation/subnmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/add_2_grad/Reshape"/device:GPU:0*
transpose_b( *
T0*
transpose_a(
?
jmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/ShapeShape:main_level/agent/main/target/network_0/observation/truediv"/device:GPU:0*
T0*
out_type0
?
lmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Shape_1Const"/device:GPU:0*
valueB *
dtype0
?
zmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/BroadcastGradientArgsBroadcastGradientArgsjmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Shapelmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Shape_1"/device:GPU:0*
T0
?
hmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/SumSumnmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/MatMul_grad/MatMulzmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
lmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/ReshapeReshapehmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Sumjmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Shape"/device:GPU:0*
Tshape0*
T0
?
jmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Sum_1Sumnmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/MatMul_grad/MatMul|main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
hmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/NegNegjmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Sum_1"/device:GPU:0*
T0
?
nmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Reshape_1Reshapehmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Neglmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
nmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/ShapeShape>main_level/agent/main/target/network_0/observation/observation"/device:GPU:0*
T0*
out_type0
?
pmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/Shape_1Const"/device:GPU:0*
valueB *
dtype0
?
~main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsnmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/Shapepmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/Shape_1"/device:GPU:0*
T0
?
pmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/RealDivRealDivlmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Reshape<main_level/agent/main/target/network_0/observation/truediv/y"/device:GPU:0*
T0
?
lmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/SumSumpmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/RealDiv~main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/BroadcastGradientArgs"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
pmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/ReshapeReshapelmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/Sumnmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
lmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/NegNeg>main_level/agent/main/target/network_0/observation/observation"/device:GPU:0*
T0
?
rmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/RealDiv_1RealDivlmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/Neg<main_level/agent/main/target/network_0/observation/truediv/y"/device:GPU:0*
T0
?
rmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/RealDiv_2RealDivrmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/RealDiv_1<main_level/agent/main/target/network_0/observation/truediv/y"/device:GPU:0*
T0
?
lmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/mulMullmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/sub_grad/Reshapermain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/RealDiv_2"/device:GPU:0*
T0
?
nmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/Sum_1Sumlmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/mul?main_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
rmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/Reshape_1Reshapenmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/Sum_1pmain_level/agent/main/target/gradients_1/main_level/agent/main/target/network_0/observation/truediv_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
4main_level/agent/main/target/output_gradient_weightsPlaceholder"/device:GPU:0*
dtype0* 
shape:?????????3
?
2main_level/agent/main/target/gradients_2/grad_ys_0Identity4main_level/agent/main/target/output_gradient_weights"/device:GPU:0*
T0
?
xmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/mulMul2main_level/agent/main/target/gradients_2/grad_ys_0Fmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/Sum/reduction_indicesConst"/device:GPU:0*
valueB :
?????????*
dtype0
?
xmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/SumSumxmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/mul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/Sum/reduction_indices"/device:GPU:0*
T0*

Tidx0*
	keep_dims(
?
xmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/subSub2main_level/agent/main/target/gradients_2/grad_ys_0xmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/Sum"/device:GPU:0*
T0
?
zmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/mul_1Mulxmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/subFmain_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax"/device:GPU:0*
T0
?
ymain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/ShapeShapeUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDims"/device:GPU:0*
T0*
out_type0
?
{main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Shape_1ShapeSmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/BroadcastGradientArgsBroadcastGradientArgsymain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Shape{main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Shape_1"/device:GPU:0*
T0
?
wmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/SumSumzmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/mul_1?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/BroadcastGradientArgs"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
{main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/ReshapeReshapewmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Sumymain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
ymain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Sum_1Sumzmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/Softmax_grad/mul_1?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
}main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Reshape_1Reshapeymain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Sum_1{main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/ShapeShapePmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/ReshapeReshape{main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Reshape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/ShapeShapeWmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape_1ShapeTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/SumSum}main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Reshape_1?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/ReshapeReshape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Sum?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Sum_1Sum}main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/output_grad/Reshape_1?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/NegNeg?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Sum_1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape_1Reshape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Neg?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/ShapeShapeSmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_1"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape_1Const"/device:GPU:0*
valueB:3*
dtype0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/SumSum?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/Reshape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/ReshapeReshape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Sum?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Sum_1Sum?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/ExpandDims_grad/Reshape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape_1Reshape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Sum_1?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ShapeShapeWmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/SizeConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :*
dtype0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/addAddfmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean/reduction_indices?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Size"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/modFloorMod?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/add?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Size"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_1Const"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
valueB *
dtype0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/startConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B : *
dtype0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/deltaConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :*
dtype0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/rangeRange?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/start?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Size?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range/delta"/device:GPU:0*

Tidx0*?
_class?
??loc:@main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Fill/valueConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :*
dtype0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/FillFill?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_1?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Fill/value"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*

index_type0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/DynamicStitchDynamicStitch?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/range?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/mod?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Fill"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
N
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum/yConst"/device:GPU:0*?
_class?
??loc:@main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape*
value	B :*
dtype0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/MaximumMaximum?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/DynamicStitch?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum/y"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordivFloorDiv?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ReshapeReshape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape_1?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/DynamicStitch"/device:GPU:0*
Tshape0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/TileTile?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Reshape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordiv"/device:GPU:0*

Tmultiples0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_2ShapeWmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_3ShapeTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean"/device:GPU:0*
out_type0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ConstConst"/device:GPU:0*
dtype0*
valueB: 
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/ProdProd?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_2?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Const_1Const"/device:GPU:0*
valueB: *
dtype0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Prod_1Prod?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Shape_3?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Const_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1/yConst"/device:GPU:0*
value	B :*
dtype0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1Maximum?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Prod_1?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1/y"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordiv_1FloorDiv?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Prod?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Maximum_1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/CastCast?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/floordiv_1"/device:GPU:0*

SrcT0*
Truncate( *

DstT0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/truedivRealDiv?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Tile?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/Cast"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMulMatMul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/ReshapePmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_4"/device:GPU:0*
T0*
transpose_a( *
transpose_b(
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMul_1MatMulOmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/Relu?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape"/device:GPU:0*
T0*
transpose_a(*
transpose_b( 
?
-main_level/agent/main/target/gradients_2/AddNAddN?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Mean_grad/truediv"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/sub_grad/Reshape*
N
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/ShapeShapeUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/ReshapeReshape-main_level/agent/main/target/gradients_2/AddN?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/Relu_grad/ReluGradReluGrad?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMulOmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/Relu"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_2_grad/MulMul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape_1Rmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/mul"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_2_grad/Mul_1Mul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_5_grad/Reshape_1_main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/ShapeShapeXmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape_1Const"/device:GPU:0*
valueB:?*
dtype0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/SumSum?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/Reshape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/ReshapeReshape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Sum?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Sum_1Sum?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Reshape_grad/Reshape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape_1Reshape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Sum_1?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Shape_1"/device:GPU:0*
Tshape0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/ShapeShapeQmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape_1Const"/device:GPU:0*
valueB:?*
dtype0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/SumSum?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/Relu_grad/ReluGrad?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/ReshapeReshape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Sum?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Sum_1Sum?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/Relu_grad/ReluGrad?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape_1Reshape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Sum_1?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_3_grad/MulMul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMul_1Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/MatMul"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_3_grad/Mul_1Mul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_1_grad/MatMul_1amain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMulMatMul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/ReshapeUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_4"/device:GPU:0*
transpose_a( *
transpose_b(*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMul_1MatMulTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Relu?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape"/device:GPU:0*
T0*
transpose_a(*
transpose_b( 
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMulMatMul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/ReshapePmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_1"/device:GPU:0*
T0*
transpose_a( *
transpose_b(
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul_1MatMulLmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape"/device:GPU:0*
transpose_a(*
transpose_b( *
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Relu_grad/ReluGradReluGrad?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMulTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Relu"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_2_grad/MulMul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape_1Wmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/mul"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_2_grad/Mul_1Mul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_5_grad/Reshape_1dmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_grad/MulMul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape_1Rmain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/mul"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_grad/Mul_1Mul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/add_2_grad/Reshape_1_main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/ShapeShapeVmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape_1Const"/device:GPU:0*
valueB:?*
dtype0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape_1"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/SumSum?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Relu_grad/ReluGrad?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/BroadcastGradientArgs"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/ReshapeReshape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Sum?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Sum_1Sum?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/Relu_grad/ReluGrad?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape_1Reshape?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Sum_1?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_3_grad/MulMul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMul_1Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/MatMul"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_3_grad/Mul_1Mul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_1_grad/MatMul_1fmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_1_grad/MulMul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul_1Umain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/MatMul"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/mul_1_grad/Mul_1Mul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul_1amain_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/read"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMulMatMul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/ReshapeUmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_1"/device:GPU:0*
transpose_b(*
T0*
transpose_a( 
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul_1MatMulLmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape"/device:GPU:0*
transpose_a(*
transpose_b( *
T0
?
/main_level/agent/main/target/gradients_2/AddN_1AddN?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul"/device:GPU:0*
T0*?
_class?
??loc:@main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/MatMul_grad/MatMul*
N
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice_grad/ShapeShape*main_level/agent/main/target/network_0/add"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice_grad/StridedSliceGradStridedSliceGrad?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice_grad/ShapeRmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice/stackTmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice/stack_1Tmain_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice/stack_2/main_level/agent/main/target/gradients_2/AddN_1"/device:GPU:0*
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask 
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_grad/MulMul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape_1Wmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/mul"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_grad/Mul_1Mul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/add_2_grad/Reshape_1dmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/read"/device:GPU:0*
T0
?
^main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/add_grad/ShapeShape*main_level/agent/main/target/network_0/mul"/device:GPU:0*
T0*
out_type0
?
`main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/add_grad/Shape_1Shape,main_level/agent/main/target/network_0/mul_1"/device:GPU:0*
T0*
out_type0
?
nmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/add_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/add_grad/Shape`main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/add_grad/Shape_1"/device:GPU:0*
T0
?
\main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/add_grad/SumSum?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice_grad/StridedSliceGradnmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
`main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/add_grad/ReshapeReshape\main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/add_grad/Sum^main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/add_grad/Shape"/device:GPU:0*
Tshape0*
T0
?
^main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/add_grad/Sum_1Sum?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/strided_slice_grad/StridedSliceGradpmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/add_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
bmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/add_grad/Reshape_1Reshape^main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/add_grad/Sum_1`main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_1_grad/MulMul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul_1Zmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/MatMul"/device:GPU:0*
T0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/mul_1_grad/Mul_1Mul?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/MatMul_grad/MatMul_1fmain_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/read"/device:GPU:0*
T0
?
^main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_grad/ShapeConst"/device:GPU:0*
valueB *
dtype0
?
`main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_grad/Shape_1Shape3main_level/agent/main/target/network_0/StopGradient"/device:GPU:0*
out_type0*
T0
?
nmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_grad/BroadcastGradientArgsBroadcastGradientArgs^main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_grad/Shape`main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_grad/Shape_1"/device:GPU:0*
T0
?
\main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_grad/MulMul`main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/add_grad/Reshape3main_level/agent/main/target/network_0/StopGradient"/device:GPU:0*
T0
?
\main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_grad/SumSum\main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_grad/Mulnmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
`main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_grad/ReshapeReshape\main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_grad/Sum^main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
^main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_grad/Mul_1Mul*main_level/agent/main/target/network_0/sub`main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/add_grad/Reshape"/device:GPU:0*
T0
?
^main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_grad/Sum_1Sum^main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_grad/Mul_1pmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
bmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_grad/Reshape_1Reshape^main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_grad/Sum_1`main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
`main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/ShapeConst"/device:GPU:0*
dtype0*
valueB 
?
bmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/Shape_1Shape.main_level/agent/main/target/network_0/mul_1/y"/device:GPU:0*
out_type0*
T0
?
pmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs`main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/Shapebmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0
?
^main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/MulMulbmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/add_grad/Reshape_1.main_level/agent/main/target/network_0/mul_1/y"/device:GPU:0*
T0
?
^main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/SumSum^main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/Mulpmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
bmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/ReshapeReshape^main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/Sum`main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
`main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/Mul_1MulMmain_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers/readbmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/add_grad/Reshape_1"/device:GPU:0*
T0
?
`main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/Sum_1Sum`main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/Mul_1rmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
dmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/Reshape_1Reshape`main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/Sum_1bmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
\main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/sub_grad/NegNeg`main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_grad/Reshape"/device:GPU:0*
T0
?
dmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1/y_grad/unstackUnpackdmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/Reshape_1"/device:GPU:0*
T0*	
num*

axis 
?
/main_level/agent/main/target/gradients_2/AddN_2AddNbmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/Reshape\main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/sub_grad/Neg"/device:GPU:0*
T0*u
_classk
igloc:@main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1_grad/Reshape*
N
?
~main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/ShapeShapeZmain_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0*
out_type0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/ReshapeReshapedmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/mul_1/y_grad/unstack~main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation_grad/ReluGradReluGrad?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/Flatten/flatten/Reshape_grad/ReshapeZmain_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation"/device:GPU:0*
T0
?
lmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/add_2_grad/ShapeShape9main_level/agent/main/target/network_0/observation/MatMul"/device:GPU:0*
T0*
out_type0
?
nmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/add_2_grad/Shape_1Const"/device:GPU:0*
valueB:?*
dtype0
?
|main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/add_2_grad/BroadcastGradientArgsBroadcastGradientArgslmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/add_2_grad/Shapenmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/add_2_grad/Shape_1"/device:GPU:0*
T0
?
jmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/add_2_grad/SumSum?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation_grad/ReluGrad|main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/add_2_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0
?
nmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/add_2_grad/ReshapeReshapejmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/add_2_grad/Sumlmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/add_2_grad/Shape"/device:GPU:0*
T0*
Tshape0
?
lmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/add_2_grad/Sum_1Sum?main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/BatchnormActivationDropout_1_activation_grad/ReluGrad~main_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/add_2_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*

Tidx0*
	keep_dims( 
?
pmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/add_2_grad/Reshape_1Reshapelmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/add_2_grad/Sum_1nmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/add_2_grad/Shape_1"/device:GPU:0*
T0*
Tshape0
?
nmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/MatMul_grad/MatMulMatMulnmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/add_2_grad/Reshape8main_level/agent/main/target/network_0/observation/add_1"/device:GPU:0*
T0*
transpose_a( *
transpose_b(
?
pmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/MatMul_grad/MatMul_1MatMul6main_level/agent/main/target/network_0/observation/subnmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/add_2_grad/Reshape"/device:GPU:0*
transpose_a(*
transpose_b( *
T0
?
hmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/mul_grad/MulMulpmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/add_2_grad/Reshape_1Fmain_level/agent/main/target/network_0/observation/NoisyNetDense_0/mul"/device:GPU:0*
T0
?
jmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/mul_grad/Mul_1Mulpmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/add_2_grad/Reshape_1Smain_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev/read"/device:GPU:0*
T0
?
jmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/mul_1_grad/MulMulpmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/MatMul_grad/MatMul_1Imain_level/agent/main/target/network_0/observation/NoisyNetDense_0/MatMul"/device:GPU:0*
T0
?
lmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/mul_1_grad/Mul_1Mulpmain_level/agent/main/target/gradients_2/main_level/agent/main/target/network_0/observation/MatMul_grad/MatMul_1Umain_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev/read"/device:GPU:0*
T0
e
,main_level/agent/main/target/AssignAdd/valueConst"/device:GPU:0*
value	B	 R*
dtype0	
?
&main_level/agent/main/target/AssignAdd	AssignAdd(main_level/agent/main/online/global_step,main_level/agent/main/target/AssignAdd/value"/device:GPU:0*
use_locking( *
T0	*;
_class1
/-loc:@main_level/agent/main/online/global_step
?N
!main_level/agent/main/target/initNoOp0^main_level/agent/main/online/beta1_power/Assign0^main_level/agent/main/online/beta2_power/Assign0^main_level/agent/main/online/global_step/Assignr^main_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam/Assignt^main_level/agent/main/online/main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/Adam_1/Assignv^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Adam/Assignx^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/Adam_1/Assignx^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Adam/Assignz^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/Adam_1/Assignx^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam/Assignz^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/Adam_1/Assignz^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam/Assign|^main_level/agent/main/online/main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Adam_1/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam/Assign?^main_level/agent/main/online/main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Adam_1/AssignP^main_level/agent/main/online/network_0/gradients_from_head_0-0_rescalers/AssignT^main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_mean/AssignV^main_level/agent/main/online/network_0/observation/NoisyNetDense_0/bias_stddev/AssignV^main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_mean/AssignX^main_level/agent/main/online/network_0/observation/NoisyNetDense_0/weight_stddev/Assigne^main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Assigng^main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Assigng^main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Assigni^main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Assigne^main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Assigng^main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Assigng^main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Assigni^main_level/agent/main/online/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Assign`^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Assignb^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Assignb^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Assignd^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Assign`^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Assignb^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Assignb^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Assignd^main_level/agent/main/online/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/AssignP^main_level/agent/main/target/network_0/gradients_from_head_0-0_rescalers/AssignT^main_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_mean/AssignV^main_level/agent/main/target/network_0/observation/NoisyNetDense_0/bias_stddev/AssignV^main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_mean/AssignX^main_level/agent/main/target/network_0/observation/NoisyNetDense_0/weight_stddev/Assigne^main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_mean/Assigng^main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/bias_stddev/Assigng^main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_mean/Assigni^main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc1/weight_stddev/Assigne^main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_mean/Assigng^main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/bias_stddev/Assigng^main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_mean/Assigni^main_level/agent/main/target/network_0/rainbow_q_values_head_0/action_advantage/fc2/weight_stddev/Assign`^main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_mean/Assignb^main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/bias_stddev/Assignb^main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_mean/Assignd^main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc1/weight_stddev/Assign`^main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_mean/Assignb^main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/bias_stddev/Assignb^main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_mean/Assignd^main_level/agent/main/target/network_0/rainbow_q_values_head_0/state_value/fc2/weight_stddev/Assign"/device:GPU:0
?
#main_level/agent/main/target/init_1NoOp-^main_level/agent/main/online/Variable/Assign7^main_level/agent/main/online/network_0/Variable/Assign-^main_level/agent/main/target/Variable/Assign7^main_level/agent/main/target/network_0/Variable/Assign"/device:GPU:0
?
'main_level/agent/main/target/group_depsNoOp"^main_level/agent/main/target/init$^main_level/agent/main/target/init_1"/device:GPU:0"