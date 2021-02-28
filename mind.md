class List(object):
pass#学长
题目描述：
有n个物品，他们有各自的体积和价值，现有给定容量的背包，如何让背包里装入的物品具有最大的价值总和?
输入参数：N和W分别是物体的数量和背包能装的重量。wt数组是指物体的重量，val指的是对应的价值

输入样例：3，4，[2,1,3],[4,2,3]

思路：     
    1.首先，这显然是一个背包问题。          
    2.其次，只有重量一个影响因素，构造二维dp即可        
3.遍历时，装与不装取最优 
        当前第i个物品背包空间j能装下时（取优）： 
dp[i][j]=max(dp[i][j],dp[i-1][j-wt[i]]+val[i])
当前第i个物品背包空间j不能装下时：
dp[i][j]=dp[i-1][j]

````python
#encoding=utf-8
def fun(n,w,wt,val):
    if not wt or not val:
        return 0
    dp=[[0 for _ in range(w+1)]for _ in range(n+1)]
    for i in range(1,n+1):
        for j in range(1,w+1):
            dp[i][j]=dp[i-1][j]#注意=dp[i-1][j],而不是[i-1][j-1]
            if(j>=wt[i-1]):
                dp[i][j] = max(dp[i][j], dp[i - 1][j - wt[i-1]] + val[i-1])
    return dp[-1][-1]
print( fun(3,4,[2,1,3],[4,2,3]))
````
### 300 最长递增子序列
分类：_动态规划_   

题目：给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

示例 ：    
输入：nums = [10,9,2,5,3,7,101,18]     
输出：4    
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/longest-increasing-subsequence              

```python
class Solution:
    def lengthOfLIS(self, nums):
        dp =[1 for i in range(len(nums))]
        for i in range(len(nums)):
            for j in range(0,i):
                if(nums[i]>nums[j]):
                    dp[i]=max(dp[j]+1,dp[i])
        return max(dp)
```

___


###一和零
````python
class Solution(object):
    def findMaxForm(self, strs, m, n):
        dp = [[[0] * (n + 1) for _ in range(m + 1)] for _ in range(len(strs) + 1)]
        for i in range(1, len(strs) + 1):
            ones = strs[i - 1].count("1")
            zeros = strs[i - 1].count("0")
            for j in range(m + 1):
                for k in range(n + 1):
                    if j >= zeros and k >= ones :
                        dp[i][j][k] = max( dp[i-1][j][k],dp[i - 1][j - zeros][k - ones] + 1)
                    else:
                        dp[i][j][k] = dp[i - 1][j][k]
        return dp[-1][-1][-1]
````
###322 零钱兑换
背包问题    
给定不同面额的硬币 coins 和一个总金额 amount。
编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币
组合能组成总金额，返回-1。

你可以认为每种硬币的数量是无限的。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/coin-change

```python
def coinChange(coins,amount):
    dp = [[1000 for _ in range(amount + 1)] for _ in range(len(coins) + 1)]
    for i in range(len(coins) + 1):
        dp[i][0] = 0
    for i in range(1,len(coins)+1):
        for j in range(1,amount+1):
            if(j>=coins[i-1]):
                dp[i][j]=min(dp[i-1][j],dp[i][j-coins[i-1]]+1)
            else:
                dp[i][j]=dp[i-1][j]
    if dp[-1][-1]==1000 :
        return -1
    else:
        return dp[-1][-1]
print( coinChange([2,5,1],11))
```
###76 最小覆盖子串
给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。

注意：如果 s 中存在这样的子串，我们保证它是唯一的答案。

来源：力扣（LeetCode）     
链接：https://leetcode-cn.com/problems/minimum-window-substring
```python
#动规--超时
def fun(s,t):
    res=""
    def record(t):
        num=len(t)
        temp = [['' for _ in range(num)], [1 for _ in range(num)]]
        j=0
        for i in range(num):
            if j==0 or t[i] not in temp[0]:
                temp[0][j]=t[i]
                for k in range(t.count(t[i])):
                    temp[1][j]-=1
            j+=1
        return temp

    temp = record(t)
    temp1=temp[1][:]
    for i in range(len(s)):
        temp[1]=temp1[:]
        for j in range(i,-1,-1):
            if s[j] in t:
                temp[1][temp[0].index(s[j])]+=1
            if min(temp[1])==1: #每个字符都找到了
                if len(res)==0 or len(res)>i-j+1:#当前res较长
                    res=s[j:i+1]#左闭右开
    if(min(temp[1])==0):
        return ""#一次符合的也没有
    return res
print( fun("ADOBECODEBANC", "ABC"))


#滑动窗口
from collections import Counter
def fun(s,t):
    windows=dict()
    needs=dict(Counter(t[:]))
    right=left=0
    volid=0
    res=float("inf")
    start=0
    while right<len(s):
        c=s[right]
        right+=1
        if c in needs:
            if c in windows:
                windows[c]+=1
            else:
                windows[c]=1
            if windows[c]==needs[c]:
                volid+=1
            while len(needs)==volid:
                d=s[left]
                left+=1
                if right-left+1<res:#这里不太一样
                    start=left-1
                    res=right-left+1
                if d in needs:
                    if windows[d]==needs[d]:
                        volid-=1
                    windows[d]-=1
    if res==float("inf"):
        return ""
    return s[start:res+start]
# print fun("ab", "a")
print( fun("ADOBECODEBANC",  "ABC"))
# # print fun("aab", "aab")
```

###字符串的排列
####滑动窗口
思路：左右两个下标围成一小块、包围一小块窗口，右下标一直往后移动，直到包含所有条件，当满足条件时，才移动左下标，来优化结果，移动时要判断左下标右移后要删掉的那个元素是否是题目所需，如果是还要进行一系列操作。
```python
def checkInclusion(s1, s2) :
    windows = dict()
    needs = dict()#不要忘记加括号
    for i in s1:
        if i in needs:
            needs[i] = needs[i] + 1
        else:
            needs[i] = 1
    right = left = 0
    voiad = 0
    while right < len(s2):
        c = s2[right]
        right += 1
        if c in needs:
            if c in windows:
                windows[c] = windows[c] + 1
            else:
                windows[c] = 1
            if windows[c] == needs[c]:
                voiad += 1
            while voiad == len(needs):
                d = s2[left]
                if right - left == len(s1):
                    return True
                if d in needs:
                    if windows[d] == needs[d]:
                        voiad -= 1
                    windows[d] = windows[d] - 1
                left = left + 1
    return False
```
###438找到字符串中所有字母异位词
####一遍过，嘻嘻
```python
from collections import  Counter
def fun(s,p):
    windows=dict()
    needs = dict(Counter(p[:]))
    right=left=0
    voild=0
    res=[]
    while right<len(s):
        c=s[right]
        right+=1
        if c in needs:
            if c in windows:
                windows[c]+=1
            else:
                windows[c]=1
            if windows[c]==needs[c]:
                voild+=1
            while voild==len(needs):
                d=s[left]
                left+=1
                if right-left+1==len(p):
                    res.append(left-1)
                if d in needs:
                    if windows[d]==needs[d]:
                        voild-=1
                    windows[d]-=1
    return res
print( fun("cbaebabacd" ,"abc"))
```
###198 [打家劫舍](https://leetcode-cn.com/problems/house-robber)
####你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
####给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
20/1/22
```python
def rob( nums):
    if not nums:
        return 0
    if len(nums)==1:
        return nums[0]
    dp=[[] for i in range(len(nums)+1)]
    dp[0]=0
    dp[1]=nums[0]
    for i in range(2,len(nums)+1):
        dp[i]=max(dp[i-1],dp[i-2]+nums[i-1])
    return dp[-1]
print(rob([1,2,3,1]))
```

###达达：72 编辑距离 [达达题解](https://www.jianshu.com/p/02561842fedd)
```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        #创建二维dp
        m = len(word1)
        n = len(word2)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(n+1):
            dp[i][0] = i
        for j in range(m+1):
            dp[0][j] = j
        for i in range(1,n+1):
            for j in range(1,m+1):
                if word1[j-1]!=word2[i-1]:
                    dp[i][j] = min(dp[i][j-1],dp[i-1][j],dp[i-1][j-1])+1
                else:
                    dp[i][j] = dp[i-1][j-1]
        return dp[-1][-1]
```
###48 学长 旋转图像
```python
from typing import List#没有的话，下文参数中出现List会飘红
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        pos1,pos2=0,(len(matrix[0])-1)      
        while pos1<pos2:
            add=0
            while add<pos2-pos1:    
                temp=matrix[pos1][pos1+add]
                matrix[pos1][pos1+add]=matrix[pos2-add][pos1]
                matrix[pos2-add][pos1]=matrix[pos2][pos2-add]
                matrix[pos2][pos2-add]=matrix[pos1+add][pos2]
                matrix[pos1+add][pos2]=temp
                add+=1
            pos1+=1
            pos2-=1
```