说到svm, 按套路就要先说说线性分类器, 如图, 在特征维数为2时, 可以用一条线将正负样本分离开来.当然了, 这条线可以有无数条, 假设我们训练得到了L2, 而L1是真正的那条直线, 对于新的测试样本(虚线的x), 显然, 用L2分类就会出现误分类. 也就是说, 线性分类器的效果并不怎么好, 但是很多都会把它作为概念的引入课程

![img](https://img-blog.csdn.net/20160829090028758)

后来92年有人提出了用一对平行的边界(boundry)来将正负样本(Pos/Neg example)分开, 其中中线确定了超平面(hyperplane)的位置, 如图. 

![img](https://img-blog.csdn.net/20160829090203041)

两个边界的距离, 我们称之为margin. 我们的目的是, 让这个margin尽可能的大, 最大边界上的正负样本, 我们称它们为支持向量(support vector). 所以如图, 对于垂直于超平面的单位向量**w**, 以及某个正样本的支持向量**u**,** u**在**w**上的投影便是右上的超平面到原点的距离, 即 **w**ᐧ**u**

**![img](https://img-blog.csdn.net/20160829161853145)**

可见正样本都是分布在**w**ᐧ**u**>= c 的区域(大于某个距离的区域), c是某个定常数. 令c = -b, 公式改写成

​                 **w**ᐧ**u**+ b >= 0                           (decision rule)

所以对任意的正样本**x**, 我们令

**                ****w****ᐧx**** **+ b >= 1

同理, 对于负样本**x**

​                **w****ᐧx**** **+ b <= - 1

相应的, 对于正负样本的标签, 分别是 y = 1 与 y=-1

这样不论对于正样本还是负样本,  我们都有

​                y(**w****ᐧx**** **+ b) >= 1

变形

​                y(**w****ᐧx**** **+ b) - 1>= 0

对于在边界上的正负样本, 有

​                y(**w****ᐧx**** **+ b) - 1 = 0

如图, 对于正负两个支持向量, 作差可以得到连接两个边界的一个向量, 再点乘前面的单位向量w, 得到了该向量在w方向上的投影, 便得到了margin的大小

![img](https://img-blog.csdn.net/20160829090046071)

![img](https://img-blog.csdn.net/20160829090344970)

到这里, 想想为什么要||w||的最小二乘方?而不是一次, 四次方?    ( cs229 Andrew的提问)

最小二乘法在很多假设下都有意义(make sense)  (Andrew的回答)

![img](https://img-blog.csdn.net/20160829111024760)

问题转化为如图形式, 这是一个凸二次规划问题(convex [quadratic programming](https://en.wikipedia.org/wiki/Quadratic_programming)), 具体什么是凸二次规划问题, 可以参考<<统计学习方法>> 100页, 该页还有最大间隔存在的唯一性的证明

在这个形式下, 就是在y(wx+b)>=1的条件下最小化 ||w||的平方, 其中以w,b为变量

将它作为原始最优化问题, 应用拉格朗日对偶性(Lagrange Duality), 通过求对偶问题(dual problem)得到原始问题的最优解[1]

其实我也是从这里第一次接触二次规划的概念, 在没有不等式的条件下, 形式和我们高数学过的多元函数求简单极值一样, 即是求闭区域内连续有界多元函数的驻点或偏导不存在点

条件极值下, 就要引入拉格朗日乘子, 印象求解过程很麻烦

**求对偶问题**

****

如下构造拉格朗日方程 L(w,b,α),  引入了拉格朗日乘子α

α)

![img](https://img-blog.csdn.net/20160829111100619)

![img](https://img-blog.csdn.net/20160829115802186)

先对w, b求极小值, 再对α求极大值, 即

![img](https://img-blog.csdn.net/20160829120654378)

对α求最大值可以转化为对下式求最小值

![img](https://img-blog.csdn.net/20160829120829622)

以上

原始问题

![img](https://img-blog.csdn.net/20160829111024760)

转化为对偶问题(对第二行式子请自行脑补min(α)   )

![img](https://img-blog.csdn.net/20160829120654378)

\---------------------------------------------------------------------------------------------------------------------------------------

**对偶问题求解步骤**

![img](https://img-blog.csdn.net/20160829120829622)

我们先根据上面的式子, 求出一组最优解 α , 后面说怎么求这个α

然后代入下面这里的 2式 求出w

![img](https://img-blog.csdn.net/20160829111100619)

因为任意支持向量满足, 所以b

![img](https://img-blog.csdn.net/20160829134625972)

实际上前面得到了w,        b = y - wx, 用任意一个支持向量去求b就可以

当我们求出了这样的一组α, w, b

代入最开始的决策函数decision rule 

**                 w**ᐧ**u**+ b >= 0                           (decision rule)

![img](https://img-blog.csdn.net/20160829130712264)

对于某个样本**u, **当**w**ᐧ**u**+ b >= 0 , 我们可以判断它为**正样本**(实际上x也是向量, 这些样本自始至终都被当作向量对待)

**回到上面的问题, 怎样求解α? **

之前我们分别先对w,b求偏导, 得到驻点, 注意这里并没有去判断是极大值点还是极小值点, 而是直接代入原方程求它对α的极大值. 

因为对b求偏导得到了等式条件3, 即所有αᐧ y之和为0, 这样我们可以减少对其中某个α的讨论. 

要想得到极值点就要分别对各个α进行求偏导, 得到驻点, 然后讨论每一个驻点以及边界点, 得到使该式子为min(对未变形的式子, 为max)的α值  

可能说的不太清楚, 大家可以看看[1]中的例题7.2, 让你自己求解一个支持向量机

\------------

上面求解过程很麻烦, 样本容量很大时, 就需要优化了. 这一部分要用到SMO算法(因为书上[1]只讲到这一个, 其他的优化算法我也不知道).

首先要知道什么是KKT,[关于KKT的ppt](http://www2.imm.dtu.dk/courses/02711/lecture3.pdf)点这里, 这个ppt对我这个非数学专业的人比较好理解, 即便是对于没有学过多元函数求最值的人, 从求无条件极值, 到恒等式条件极值, 到等式/不等式条件极值循序渐进的介绍, 值得一看

所谓kkt条件, 就是最下面那三行式子

![img](https://img-blog.csdn.net/20160830221836249)

直观一点去对应

![img](https://img-blog.csdn.net/20160830224228094)

这不就是我前面推的那个式子么![大笑](http://static.blog.csdn.net/xheditor/xheditor_emot/default/laugh.gif), 但是[1]基本上没提kkt是什么, wiki介绍的也不好, 这个地方写给和我一样曾有疑问的同学看.

smo的思路: 

\1. 如果所有变量的解都满足KKT, 则该最优化问题的解得到

\2. 如果不满足KKT, 则先选择两个变量, 固定其他变量, 对这两个变量构建二次规划问题

说句比较挨揍的话, 细节请看书吧, 后面的代码实现会再接触这些细节

![img](https://img-blog.csdn.net/20160830100126617)

![img](https://img-blog.csdn.net/20160830230354718)

图片来源:[1]

\-------------------------------------------------------------------------------------------------------------------

libsvm代码分析:

看源码之前还要看看关于核函数和松弛变量的部分, 暂时先不讲了

先从简单的说起吧

**1. decision function与predict函数**

svm得到的decision function结构如下 ,  其中f.alpha = alpha, f.rho = si.rho  

alpha与rho都是训练得到的, rho实际上就是截距, 也就是决策函数 y = **w**ᐧ**u**+ b 中的 -b         

**[cpp]** [view plain](https://blog.csdn.net/traumland/article/details/52343924?locationNum=9&fps=1#) [copy](https://blog.csdn.net/traumland/article/details/52343924?locationNum=9&fps=1#)

1. struct decision_function  
2. {  
3. ​    double *alpha;  
4. ​    double rho;  
5. };  

得到判断函数后, 就可以拿他来predict

对应函数为

**[cpp]** [view plain](https://blog.csdn.net/traumland/article/details/52343924?locationNum=9&fps=1#) [copy](https://blog.csdn.net/traumland/article/details/52343924?locationNum=9&fps=1#)

1. model=svm_load_model(argv[i+1]);  
2. x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));  
3. predict_label = svm_predict(model,x);  
4. double svm_predict(const svm_model *model, const svm_node *x)  
5. svm_predict_values(model, x, dec_values);  
6. double svm_predict_values(const svm_model *model, const svm_node *x, double* dec_values);  

我目前只关心线性svm分类器,即c_svc, 线性核

int nr_class = model->nr_class;     //类别数
int l = model->l;                              //支持向量总数
double *kvalue = Malloc(double,l);          //#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
for(i=0;i<l;i++)
​    kvalue[i] = Kernel::k_function(x,model->SV[i],model->param);
// k_function按照kernel_type选择不同的返回值, 对线性核函数, 返回前两个前两个参数的点乘(Kernel::dot())
// model即我们训练得到的model
​    int *start = Malloc(int,nr_class);              //分配类别数大小的内存
​    start[0] = 0;                              //标记kvalue对于不同类的支持向量的起始位置, 因为kvalue是一整块内存

for(i=1;i<nr_class;i++)                            
​    start[i] = start[i-1]+model->nSV[i-1];             //nSV[i]--第i+1类的支持向量数
int *vote = Malloc(int,nr_class);              //分配类别数大小的内存用来投票
for(i=0;i<nr_class;i++)
​    vote[i] = 0;                                      //初始化投票数为0
int p=0;
for(i=0;i<nr_class;i++)
​    for(int j=i+1;j<nr_class;j++)
​    {
​        double sum = 0;
​        int si = start[i];                         //类别i对应的核函数得到的结果位置
​        int sj = start[j];                         //类别j对应的核函数得到的结果位置
​        int ci = model->nSV[i];            //类别i对应的支持向量数
​        int cj = model->nSV[j];            // 类别j对应的支持向量数
​        int k;
​        double *coef1 = model->sv_coef[j-1];           //decision function的参数 [0,nr_class)
​        double *coef2 = model->sv_coef[i];              //                                       [0,nr_class)
​        for(k=0;k<ci;k++)
​            sum += coef1[si+k] * kvalue[si+k];        
​        for(k=0;k<cj;k++)
​            sum += coef2[sj+k] * kvalue[sj+k];
​        sum -= model->rho[p];
​        dec_values[p] = sum;

\------------------------------------------------------------------------------------------------------

上面这一段对应到opencv的代码是

​                            const DecisionFunc& df = svm->decision_func[dfi];
​                            sum = -df.rho;
​                            int sv_count = svm->getSVCount(dfi);
​                            const double* alpha = &svm->df_alpha[df.ofs];
​                            const int* sv_index = &svm->df_index[df.ofs];
​                            for( k = 0; k < sv_count; k++ )
​                                sum += alpha[k]*buffer[sv_index[k]];
​                            vote[sum > 0 ? i : j]++;

在我眼里opencv的可读性更好

\----------------------------------------------------------------------------------------------------
​        if(dec_values[p] > 0)
​            ++vote[i];
​        else
​            ++vote[j];
​        p++;
​    }
​    int vote_max_idx = 0;
​    for(i=1;i<nr_class;i++)
​        if(vote[i] > vote[vote_max_idx])
​            vote_max_idx = i;
​    free(kvalue);
​    free(start);
​    free(vote);
​    return model->label[vote_max_idx];

}

有同学觉得不太明白, 不知道这张图够不够清楚, K(x,SV[i])代表 核函数K(Xi,Xj), 写的有点潦草请意会![大笑](http://static.blog.csdn.net/xheditor/xheditor_emot/default/laugh.gif)

![img](https://img-blog.csdn.net/20160902085154213)