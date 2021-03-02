import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

# matplotlib.rcParams['text.usetex'] = True


# scores_list = [[0.9951219512195122, 0.0], [0.9878048780487805, 0.008264462809917356], [0.9768292682926829, 0.024793388429752067], [0.9707317073170731, 0.04132231404958678], [0.9548780487804878, 0.04132231404958678], [0.9390243902439024, 0.049586776859504134], [0.8414634146341463, 0.6363636363636364], [0.8317073170731707, 0.6446280991735537], [0.8170731707317073, 0.6528925619834711], [0.8060975609756098, 0.6611570247933884], [0.8792682926829268, 0.08264462809917356], [0.7975609756097561, 0.19008264462809918], [0.8646341463414634, 0.09090909090909091], [0.7573170731707317, 0.19008264462809918], [0.7426829268292683, 0.6859504132231405], [0.8097560975609757, 0.09917355371900827], [0.7951219512195122, 0.10743801652892562], [0.7841463414634147, 0.10743801652892562], [0.7853658536585366, 0.10743801652892562], [0.7548780487804878, 0.10743801652892562], [0.7414634146341463, 0.11570247933884298], [0.7292682926829268, 0.14049586776859505], [0.7134146341463414, 0.15702479338842976], [0.7036585365853658, 0.19008264462809918], [0.7036585365853658, 0.21487603305785125], [0.6914634146341463, 0.24793388429752067], [0.65, 0.3140495867768595], [0.6658536585365854, 0.34710743801652894], [0.525609756097561, 0.371900826446281], [0.6390243902439025, 0.3884297520661157], [0.5621951219512196, 0.8512396694214877], [0.6109756097560975, 0.39669421487603307], [0.6170731707317073, 0.2975206611570248], [0.5878048780487805, 0.4214876033057851], [0.5146341463414634, 0.8512396694214877], [0.5646341463414634, 0.7107438016528925], [0.5609756097560976, 0.33884297520661155], [0.5329268292682927, 0.743801652892562], [0.5195121951219512, 0.7520661157024794], [0.3951219512195122, 0.371900826446281], [0.48902439024390243, 0.7520661157024794], [0.49878048780487805, 0.34710743801652894], [0.47073170731707314, 0.7933884297520661], [0.4585365853658537, 0.7933884297520661], [0.4378048780487805, 0.49586776859504134], [0.4304878048780488, 0.8181818181818182], [0.3853658536585366, 0.8842975206611571], [0.375609756097561, 0.8842975206611571], [0.39634146341463417, 0.5206611570247934], [0.35365853658536583, 0.8925619834710744], [0.35609756097560974, 0.859504132231405], [0.34146341463414637, 0.859504132231405], [0.33414634146341465, 0.859504132231405], [0.32439024390243903, 0.859504132231405], [0.32439024390243903, 0.859504132231405], [0.3121951219512195, 0.8677685950413223], [0.2865853658536585, 0.9090909090909091], [0.28292682926829266, 0.8760330578512396], [0.275609756097561, 0.8925619834710744], [0.2646341463414634, 0.8925619834710744], [0.24146341463414633, 0.9173553719008265], [0.2451219512195122, 0.9008264462809917], [0.23536585365853657, 0.9008264462809917], [0.25121951219512195, 0.8264462809917356], [0.2121951219512195, 0.9090909090909091], [0.19878048780487806, 0.9256198347107438], [0.17560975609756097, 0.9504132231404959], [0.19878048780487806, 0.9090909090909091], [0.17682926829268292, 0.9504132231404959], [0.1975609756097561, 0.6942148760330579], [0.14878048780487804, 0.9586776859504132], [0.1353658536585366, 0.9752066115702479], [0.13048780487804879, 0.9752066115702479], [0.12439024390243902, 0.9752066115702479], [0.11219512195121951, 0.9834710743801653], [0.12804878048780488, 0.9504132231404959], [0.09634146341463415, 0.8181818181818182], [0.09878048780487805, 0.9752066115702479], [0.08780487804878048, 0.9834710743801653], [0.08170731707317073, 0.9917355371900827], [0.06951219512195123, 0.9917355371900827], [0.06341463414634146, 0.9917355371900827], [0.09268292682926829, 0.9586776859504132], [0.08536585365853659, 0.9669421487603306], [0.05853658536585366, 0.9752066115702479], [0.06097560975609756, 0.9669421487603306], [0.03902439024390244, 0.9917355371900827], [0.03048780487804878, 0.9917355371900827], [0.02804878048780488, 0.9917355371900827], [0.023170731707317073, 1.0], [0.014634146341463415, 1.0], [0.007317073170731708, 1.0], [0.006097560975609756, 1.0], [0.0012195121951219512, 0.9752066115702479], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
# scores_list   = [[0.9951219512195122, 0.0], [0.9853658536585366, 0.008264462809917356], [0.973170731707317, 0.024793388429752067], [0.9658536585365853, 0.04132231404958678], [0.9536585365853658, 0.04132231404958678], [0.9390243902439024, 0.049586776859504134], [0.9280487804878049, 0.05785123966942149], [0.9195121951219513, 0.06611570247933884], [0.9060975609756098, 0.0743801652892562], [0.8939024390243903, 0.08264462809917356], [0.8780487804878049, 0.08264462809917356], [0.8634146341463415, 0.08264462809917356], [0.8524390243902439, 0.09090909090909091], [0.8353658536585366, 0.09090909090909091], [0.8195121951219512, 0.09090909090909091], [0.8097560975609757, 0.09917355371900827], [0.7951219512195122, 0.10743801652892562], [0.7865853658536586, 0.10743801652892562], [0.7719512195121951, 0.10743801652892562], [0.7585365853658537, 0.10743801652892562], [0.7414634146341463, 0.11570247933884298], [0.7292682926829268, 0.14049586776859505], [0.7195121951219512, 0.15702479338842976], [0.7085365853658536, 0.19008264462809918], [0.7024390243902439, 0.21487603305785125], [0.6939024390243902, 0.24793388429752067], [0.6829268292682927, 0.2727272727272727], [0.6695121951219513, 0.34710743801652894], [0.6560975609756098, 0.371900826446281], [0.6390243902439025, 0.3884297520661157], [0.6292682926829268, 0.3884297520661157], [0.6121951219512195, 0.39669421487603307], [0.6085365853658536, 0.4049586776859504], [0.5914634146341463, 0.4214876033057851], [0.5804878048780487, 0.6198347107438017], [0.5646341463414634, 0.7107438016528925], [0.5439024390243903, 0.7272727272727273], [0.5329268292682927, 0.743801652892562], [0.5195121951219512, 0.7520661157024794], [0.5073170731707317, 0.7520661157024794], [0.49390243902439024, 0.7520661157024794], [0.4817073170731707, 0.7768595041322314], [0.4682926829268293, 0.7933884297520661], [0.4585365853658537, 0.7933884297520661], [0.44634146341463415, 0.8099173553719008], [0.43414634146341463, 0.8181818181818182], [0.4195121951219512, 0.8181818181818182], [0.41097560975609754, 0.8181818181818182], [0.39390243902439026, 0.8264462809917356], [0.38414634146341464, 0.8512396694214877], [0.3646341463414634, 0.859504132231405], [0.3463414634146341, 0.859504132231405], [0.3353658536585366, 0.859504132231405], [0.32682926829268294, 0.859504132231405], [0.32439024390243903, 0.859504132231405], [0.3195121951219512, 0.8677685950413223], [0.3024390243902439, 0.8760330578512396], [0.28780487804878047, 0.8760330578512396], [0.2731707317073171, 0.8925619834710744], [0.2634146341463415, 0.8925619834710744], [0.25365853658536586, 0.9008264462809917], [0.24634146341463414, 0.9008264462809917], [0.23658536585365852, 0.9008264462809917], [0.22195121951219512, 0.9008264462809917], [0.2121951219512195, 0.9090909090909091], [0.19878048780487806, 0.9256198347107438], [0.1878048780487805, 0.9338842975206612], [0.18414634146341463, 0.9504132231404959], [0.17682926829268292, 0.9504132231404959], [0.16463414634146342, 0.9586776859504132], [0.14878048780487804, 0.9586776859504132], [0.1378048780487805, 0.9752066115702479], [0.1329268292682927, 0.9752066115702479], [0.12804878048780488, 0.9752066115702479], [0.12317073170731707, 0.9752066115702479], [0.11097560975609756, 0.9752066115702479], [0.1048780487804878, 0.9752066115702479], [0.1, 0.9752066115702479], [0.08902439024390243, 0.9834710743801653], [0.08292682926829269, 0.9917355371900827], [0.07195121951219512, 0.9917355371900827], [0.06463414634146342, 0.9917355371900827], [0.06341463414634146, 0.9917355371900827], [0.05975609756097561, 0.9917355371900827], [0.05731707317073171, 0.9917355371900827], [0.046341463414634146, 0.9917355371900827], [0.041463414634146344, 0.9917355371900827], [0.03170731707317073, 0.9917355371900827], [0.02926829268292683, 0.9917355371900827], [0.024390243902439025, 1.0], [0.014634146341463415, 1.0], [0.007317073170731708, 1.0], [0.006097560975609756, 1.0], [0.0024390243902439024, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
# scores_list   = [[0.9975429975429976, 0.0], [0.9815724815724816, 0.031496062992125984], [0.972972972972973, 0.07086614173228346], [0.9606879606879607, 0.09448818897637795], [0.9484029484029484, 0.10236220472440945], [0.9373464373464373, 0.10236220472440945], [0.9250614250614251, 0.10236220472440945], [0.9152334152334153, 0.11811023622047244], [0.9017199017199017, 0.12598425196850394], [0.8894348894348895, 0.12598425196850394], [0.8808353808353808, 0.13385826771653545], [0.8685503685503686, 0.13385826771653545], [0.8562653562653563, 0.14960629921259844], [0.8390663390663391, 0.16535433070866143], [0.828009828009828, 0.16535433070866143], [0.8144963144963145, 0.1732283464566929], [0.7997542997542998, 0.1732283464566929], [0.7886977886977887, 0.1889763779527559], [0.7776412776412777, 0.1889763779527559], [0.7665847665847666, 0.1889763779527559], [0.7518427518427518, 0.25984251968503935], [0.7395577395577395, 0.2677165354330709], [0.7297297297297297, 0.29133858267716534], [0.7162162162162162, 0.30708661417322836], [0.7014742014742015, 0.41732283464566927], [0.6891891891891891, 0.6062992125984252], [0.6781326781326781, 0.6377952755905512], [0.6670761670761671, 0.6614173228346457], [0.6535626535626535, 0.6850393700787402], [0.6375921375921376, 0.7007874015748031], [0.628992628992629, 0.7007874015748031], [0.6191646191646192, 0.7165354330708661], [0.6081081081081081, 0.7480314960629921], [0.5933660933660934, 0.7480314960629921], [0.5835380835380836, 0.7637795275590551], [0.5687960687960688, 0.7716535433070866], [0.5552825552825553, 0.7795275590551181], [0.5442260442260443, 0.7874015748031497], [0.5294840294840295, 0.7874015748031497], [0.5171990171990172, 0.7952755905511811], [0.5036855036855037, 0.7952755905511811], [0.4864864864864865, 0.8031496062992126], [0.47542997542997545, 0.8031496062992126], [0.4557739557739558, 0.8110236220472441], [0.4348894348894349, 0.8110236220472441], [0.4250614250614251, 0.8346456692913385], [0.41154791154791154, 0.84251968503937], [0.39926289926289926, 0.8503937007874016], [0.3882063882063882, 0.8503937007874016], [0.371007371007371, 0.8582677165354331], [0.3574938574938575, 0.8582677165354331], [0.34275184275184273, 0.8582677165354331], [0.3353808353808354, 0.8582677165354331], [0.32186732186732187, 0.8582677165354331], [0.3083538083538084, 0.8582677165354331], [0.2972972972972973, 0.8582677165354331], [0.28501228501228504, 0.8582677165354331], [0.2751842751842752, 0.8661417322834646], [0.2628992628992629, 0.8818897637795275], [0.2542997542997543, 0.8818897637795275], [0.24201474201474202, 0.889763779527559], [0.2321867321867322, 0.889763779527559], [0.22358722358722358, 0.889763779527559], [0.21621621621621623, 0.889763779527559], [0.20393120393120392, 0.8976377952755905], [0.19533169533169534, 0.9133858267716536], [0.18427518427518427, 0.9291338582677166], [0.18058968058968058, 0.9291338582677166], [0.17567567567567569, 0.937007874015748], [0.17076167076167076, 0.9448818897637795], [0.16093366093366093, 0.9606299212598425], [0.14373464373464373, 0.9606299212598425], [0.13267813267813267, 0.9606299212598425], [0.12530712530712532, 0.968503937007874], [0.11916461916461916, 0.984251968503937], [0.11056511056511056, 0.984251968503937], [0.10442260442260443, 0.9921259842519685], [0.10073710073710074, 0.9921259842519685], [0.08968058968058969, 0.9921259842519685], [0.08476658476658476, 0.9921259842519685], [0.07371007371007371, 0.9921259842519685], [0.0687960687960688, 0.9921259842519685], [0.06265356265356266, 0.9921259842519685], [0.056511056511056514, 0.9921259842519685], [0.052825552825552825, 0.9921259842519685], [0.04914004914004914, 0.9921259842519685], [0.042997542997543, 0.9921259842519685], [0.03931203931203931, 0.9921259842519685], [0.03194103194103194, 0.9921259842519685], [0.02457002457002457, 1.0], [0.0171990171990172, 1.0], [0.012285012285012284, 1.0], [0.007371007371007371, 1.0], [0.0036855036855036856, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]

#gamma=10
#scores_list   = [[1.0, 0.0], [0.9815724815724816, 0.031496062992125984], [0.9680589680589681, 0.07086614173228346], [0.9557739557739557, 0.09448818897637795], [0.9422604422604423, 0.10236220472440945], [0.9312039312039312, 0.10236220472440945], [0.9176904176904177, 0.10236220472440945], [0.9054054054054054, 0.11811023622047244], [0.8955773955773956, 0.12598425196850394], [0.8820638820638821, 0.12598425196850394], [0.8722358722358723, 0.13385826771653545], [0.85995085995086, 0.13385826771653545], [0.8488943488943489, 0.14960629921259844], [0.8316953316953317, 0.16535433070866143], [0.8218673218673219, 0.1732283464566929], [0.8095823095823096, 0.18110236220472442], [0.7972972972972973, 0.18110236220472442], [0.785012285012285, 0.1968503937007874], [0.773955773955774, 0.1968503937007874], [0.757985257985258, 0.1968503937007874], [0.7457002457002457, 0.1968503937007874], [0.7346437346437347, 0.1968503937007874], [0.7272727272727273, 0.2047244094488189], [0.7125307125307125, 0.2047244094488189], [0.6965601965601965, 0.2283464566929134], [0.6818181818181818, 0.29133858267716534], [0.6683046683046683, 0.4094488188976378], [0.6584766584766585, 0.6692913385826772], [0.6461916461916462, 0.6850393700787402], [0.6339066339066339, 0.7007874015748031], [0.6228501228501229, 0.7007874015748031], [0.6105651105651105, 0.7322834645669292], [0.597051597051597, 0.7559055118110236], [0.5835380835380836, 0.7637795275590551], [0.5712530712530712, 0.7716535433070866], [0.5577395577395577, 0.7952755905511811], [0.5417690417690417, 0.8188976377952756], [0.5319410319410319, 0.8267716535433071], [0.5196560196560197, 0.8267716535433071], [0.5061425061425061, 0.8346456692913385], [0.4975429975429975, 0.8346456692913385], [0.48525798525798525, 0.84251968503937], [0.47174447174447176, 0.84251968503937], [0.4606879606879607, 0.8503937007874016], [0.45085995085995084, 0.8582677165354331], [0.44226044226044225, 0.8661417322834646], [0.4238329238329238, 0.8661417322834646], [0.40540540540540543, 0.8740157480314961], [0.3906633906633907, 0.8818897637795275], [0.3783783783783784, 0.8976377952755905], [0.36977886977886976, 0.905511811023622], [0.3574938574938575, 0.905511811023622], [0.343980343980344, 0.905511811023622], [0.32555282555282555, 0.905511811023622], [0.3157248157248157, 0.905511811023622], [0.3034398034398034, 0.9133858267716536], [0.28746928746928746, 0.9133858267716536], [0.269041769041769, 0.9212598425196851], [0.2592137592137592, 0.9291338582677166], [0.25307125307125306, 0.9291338582677166], [0.2457002457002457, 0.9448818897637795], [0.2371007371007371, 0.9448818897637795], [0.22604422604422605, 0.9448818897637795], [0.21621621621621623, 0.9448818897637795], [0.20515970515970516, 0.9448818897637795], [0.19533169533169534, 0.9448818897637795], [0.1904176904176904, 0.968503937007874], [0.18181818181818182, 0.968503937007874], [0.17567567567567569, 0.968503937007874], [0.17076167076167076, 0.984251968503937], [0.1597051597051597, 0.984251968503937], [0.14864864864864866, 0.984251968503937], [0.13513513513513514, 0.984251968503937], [0.13022113022113022, 0.9921259842519685], [0.12162162162162163, 0.9921259842519685], [0.11056511056511056, 0.9921259842519685], [0.10810810810810811, 0.9921259842519685], [0.10319410319410319, 0.9921259842519685], [0.0945945945945946, 0.9921259842519685], [0.08845208845208845, 0.9921259842519685], [0.08108108108108109, 0.9921259842519685], [0.07493857493857493, 0.9921259842519685], [0.07002457002457002, 0.9921259842519685], [0.06142506142506143, 0.9921259842519685], [0.056511056511056514, 0.9921259842519685], [0.056511056511056514, 0.9921259842519685], [0.05036855036855037, 0.9921259842519685], [0.045454545454545456, 0.9921259842519685], [0.038083538083538086, 0.9921259842519685], [0.029484029484029485, 1.0], [0.019656019656019656, 1.0], [0.014742014742014743, 1.0], [0.012285012285012284, 1.0], [0.002457002457002457, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]

#gamma=25
scores_list = [[0.9963144963144963, 0.0], [0.9815724815724816, 0.031496062992125984], [0.9422604422604423, 0.08661417322834646], [0.9275184275184275, 0.11023622047244094], [0.9103194103194103, 0.11811023622047244], [0.8955773955773956, 0.12598425196850394], [0.8796068796068796, 0.14960629921259844], [0.8624078624078624, 0.18110236220472442], [0.8476658476658476, 0.6456692913385826], [0.8353808353808354, 0.6692913385826772], [0.8255528255528255, 0.6850393700787402], [0.812039312039312, 0.6850393700787402], [0.7960687960687961, 0.7007874015748031], [0.7751842751842751, 0.7086614173228346], [0.7653562653562653, 0.7322834645669292], [0.7518427518427518, 0.7401574803149606], [0.7383292383292384, 0.7559055118110236], [0.726044226044226, 0.7795275590551181], [0.7174447174447175, 0.7795275590551181], [0.7039312039312039, 0.7952755905511811], [0.687960687960688, 0.7952755905511811], [0.671990171990172, 0.8031496062992126], [0.6646191646191646, 0.8031496062992126], [0.6498771498771498, 0.8031496062992126], [0.6412776412776413, 0.8188976377952756], [0.6167076167076168, 0.8267716535433071], [0.6068796068796068, 0.84251968503937], [0.5958230958230958, 0.84251968503937], [0.5835380835380836, 0.84251968503937], [0.5737100737100738, 0.8503937007874016], [0.5663390663390664, 0.8503937007874016], [0.5515970515970516, 0.8582677165354331], [0.5417690417690417, 0.8582677165354331], [0.5171990171990172, 0.8582677165354331], [0.5073710073710074, 0.8661417322834646], [0.4963144963144963, 0.8740157480314961], [0.4828009828009828, 0.8740157480314961], [0.4742014742014742, 0.8740157480314961], [0.4631449631449631, 0.8740157480314961], [0.4484029484029484, 0.8740157480314961], [0.44103194103194104, 0.8740157480314961], [0.4275184275184275, 0.8818897637795275], [0.414004914004914, 0.8818897637795275], [0.4004914004914005, 0.889763779527559], [0.3894348894348894, 0.889763779527559], [0.3820638820638821, 0.8976377952755905], [0.36977886977886976, 0.8976377952755905], [0.35135135135135137, 0.8976377952755905], [0.3415233415233415, 0.905511811023622], [0.33046683046683045, 0.9212598425196851], [0.32186732186732187, 0.9291338582677166], [0.3108108108108108, 0.9291338582677166], [0.300982800982801, 0.9291338582677166], [0.28992628992628994, 0.9291338582677166], [0.28255528255528256, 0.9291338582677166], [0.2714987714987715, 0.9291338582677166], [0.2542997542997543, 0.9291338582677166], [0.24078624078624078, 0.9291338582677166], [0.2334152334152334, 0.937007874015748], [0.2285012285012285, 0.937007874015748], [0.22358722358722358, 0.952755905511811], [0.21621621621621623, 0.952755905511811], [0.20515970515970516, 0.952755905511811], [0.1977886977886978, 0.952755905511811], [0.18673218673218672, 0.952755905511811], [0.17936117936117937, 0.952755905511811], [0.1732186732186732, 0.968503937007874], [0.16339066339066338, 0.968503937007874], [0.1597051597051597, 0.968503937007874], [0.1547911547911548, 0.984251968503937], [0.14496314496314497, 0.984251968503937], [0.13513513513513514, 0.984251968503937], [0.12530712530712532, 0.984251968503937], [0.12285012285012285, 0.9921259842519685], [0.11547911547911548, 0.9921259842519685], [0.10565110565110565, 0.9921259842519685], [0.10196560196560196, 0.9921259842519685], [0.09705159705159705, 0.9921259842519685], [0.08722358722358722, 0.9921259842519685], [0.07862407862407862, 0.9921259842519685], [0.07371007371007371, 0.9921259842519685], [0.06756756756756757, 0.9921259842519685], [0.06388206388206388, 0.9921259842519685], [0.05528255528255528, 0.9921259842519685], [0.05036855036855037, 0.9921259842519685], [0.05036855036855037, 0.9921259842519685], [0.044226044226044224, 0.9921259842519685], [0.03931203931203931, 0.9921259842519685], [0.033169533169533166, 0.9921259842519685], [0.02457002457002457, 1.0], [0.014742014742014743, 1.0], [0.009828009828009828, 1.0], [0.007371007371007371, 1.0], [0.002457002457002457, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]

scores = np.array(scores_list)

x = scores[:, 0]
y = scores[:, 1]

def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]
    
    
    
pareto = identify_pareto(scores)

pareto_front = scores[pareto]

pareto_front_df = pd.DataFrame(pareto_front)
pareto_front_df.sort_values(0, inplace=True)
pareto_front = pareto_front_df.values   
    
    
x_all = scores[:, 0]
y_all = scores[:, 1]
x_pareto = pareto_front[:, 0]
y_pareto = pareto_front[:, 1]

plt.figure(figsize=(5,5))
fig,ax = plt.subplots(figsize=(5,5))
sc = plt.scatter(x_all, y_all)


annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    annot.set_text("$\gamma = "+str(scores_list.index([pos[0],pos[1]])+1)+"$")
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)
    
    
# fig.tight_layout()
plt.plot(x_pareto, y_pareto, color='r')
plt.xlabel('TPR')
plt.ylabel('TNR')


# plt.show()
# exit()


plt.savefig(('user_study/pareto.pdf'))
plt.savefig(('user_study/pareto.png'))   

    
exit()