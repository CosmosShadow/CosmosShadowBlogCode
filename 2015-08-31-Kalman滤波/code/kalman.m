clear

N=200;
w(1)=0;
w=randn(1,N);
x(1)=25;
a=1;
for k=2:N;
	x(k)=x(1)+w(k-1);
end

V=randn(1,N);
q1=std(V);
Rvv=q1.^2;

q2=std(x);
Rxx=q2.^2;

q3=std(w);
Rww = q3.^2;

p(1) = 1;
s(1) = x(1);
for t=2:N;
	p1(t) = p(t-1);				%原始生成没有噪声
	%p1(t) = p(t-1) + 0.01*Rww;	%原始少量噪声
	%p1(t) = p(t-1) + Rww;			%原始全噪声
	b(t) = p1(t) / (p1(t) + Rvv);
	s(t) = s(t-1) + b(t)*(x(t)-s(t-1));
	p(t) = (1 - b(t)) * p1(t);
end

t=1:N;
plot(t,s,'r',t,x,'b');