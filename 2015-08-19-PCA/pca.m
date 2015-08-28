clear

m = 100;
n = 3;

S = fix(rand(m, n)*50);				%随机数
S = S - repmat(mean(S), m,1);	%中心化: 均值为0
C = (S'*S)./(size(S,1)-1)			%'协方差

[P,Lambda] = eig(C);					%特征向量、特征值

S1 = S * P;	%投影在主层份上得到新变量

S1 = S1 - repmat(mean(S1), m,1);	%中心化: 均值为0
C1 = (S1'*S1)./(size(S1,1)-1)			%'新的协方差