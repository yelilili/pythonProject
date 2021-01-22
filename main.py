# coding=utf-8
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings
# See PyCharm help at https://www.jetbrains.com/help/pycharm/


# Press the green button in the gutter to run the script.#if __name__ == '__main__':

# def fun(nums):
#     list=[]
#     def digui(temp):
#         if len(temp)==len(nums):
#             list.append(temp[:])
#             return
#         for i in nums:
#             if i not in temp:
#                 temp.append(i)
#                 digui(temp)
#                 temp.pop()
#     digui([])
#     return list
# print(fun([1,2,3]))

# #全排列
# def fun(nums):
#     list=[]
#     def digui(temp):
#        # temp=[]
#         if len(temp)==len(nums):
#             list.append(temp[:])
#             return
#         for i in nums:
#             if i not in temp:
#                 temp.append(i)
#                 digui(temp)
#                 temp.pop()
#     digui([])#能不能不放参数
#     return list
# print(fun([1,2,3,4]))


# 单词搜索
# board = [["A","B","C","E"],["S","F","B","S"],["A","D","A","E"]]
# word = "ABCBA"
# def solution(board,word):
#     w=len(board)
#     h=len(board[0])
#     def compute(i,t,count):
#         if i>=w or t>=h or board[i][t]!=word[count]:
#             return False
#         if count ==len(word)-1:
#             return True
#
#         board[i][t], temp = '/', board[i][t]
#         flag = compute(i + 1, t, count + 1) or compute(i, t + 1, count + 1) or compute(i - 1, t, count + 1) or compute( i, t - 1, count + 1)#falg跳到哪
#         board[i][t] = temp
#         return flag
#     for i in range(len(board)):
#         for j in range(len(board[0])):
#             if compute(i,j,0):
#                 return True
#     return False
# print(solution(board,word))
#



# class Solution:
#     def partition(self, s):
#         def recall(s, tmp):
#             if not s:
#                 res.append(tmp[:])
#                 return
#             for i in range(1, len(s)+1):
#                 if s[:i] == s[:i][::-1]:#反转字符串
#                     tmp.append(s[:i])
#                     recall(s[i:], tmp)
#                     tmp.pop()
#         res = []
#         recall(s, [])
#         return res
# ss=Solution()
# print(ss.partition("bdaca"))


#全排列
# def fun(nums):
#     list=[]
#     def digui(temp):
#         if(len(temp)==len(nums)):#nums错写成list了
#             list.append(temp[:])
#             return
#         for i in nums:
#             if i not in temp:
#                 temp.append(i)
#                 digui(temp)  #错写成digui(i)
#                 temp.pop()
#     digui([])
#     return list
# print(fun([1,2,3]))
#
#全排列-2
# def fun(nums):
#     list=[]
#     def digui(ll):
#         if(len(ll)==len(nums)):
#             list.append(ll[:])#没写  [:]
#             return
#         for i in nums:
#             if i not in ll:
#                 ll.append(i)
#                 digui(ll)
#                 ll.pop()
#     digui([])
#     return list
# print(fun([1,2,3,4]))

#
#
# word=[['A','B','C'],['A','H','B'],['A','B','C']]
# board = 'ABHBC'
# def fun(word,board):#多加了一个参数count
#     w=len(word)
#     h=len(word[0])
#     def digui(i,j,count):
#         if i>=w or j>=h or word[i][j]!=board[count]:
#             return False
#         if count==len(board)-1:
#             return True
#         temp = word[i][j]
#         word[i][j] = '/'
#
#         flag = digui(i + 1, j, count + 1) or digui(i, j + 1, count + 1) or digui(i - 1, j, count + 1) or digui(i, j - 1,
#                                                                                                               count + 1)
#         word[i][j] = temp
#
#         return flag
#     for i in range(len(word)):
#         for j in range(len(word[0])):
#             if digui(i,j,0):
#                 return True
#     return False #忘记写这行了
# print(fun(word,board))


#
# class Solution:
#     def partition(self, s):
#         def recall(s, tmp):
#             if not s:
#                 res.append(tmp[:])
#                 return
#             for i in range(1, len(s)+1):
#                 if s[:i] == s[:i][::-1]:#反转字符串
#                     tmp.append(s[:i])
#
#                     recall(s[i:], tmp)
#                     tmp.pop()
#         res = []
#         recall(s, [])
#         return res
# ss=Solution()
# print(ss.partition("aaaa"))
#
#
# #最长递增字符串
#回溯方法
# #
# def zeng(nums):
#     for i in range(1,len(nums)):
#         if nums[i]<=nums[i-1] :
#             return False
#     return True
# def fun(nums):
#     list=[]
#     def digui(s,temp,count):
#         if  temp and not zeng(temp):
#             return
#         list.append(len(temp))
#         if not s:
#             return
#         for i in range(len(s)):
#             temp.append(s[i])
#             digui(s[i+1:],temp,count+1)
#             temp.pop()
#         return False
#     digui(nums,[],0)
#     return list
#
# print(max(fun([10,9,2,5,3,7,101,16,18])))
# print(max(fun([6,6,6,6,6,6])))
# print max(fun([0,1,0,3,2,3]))


# #最长递增字符串
#  动态规划
# def fun(nums):
#     dp=[1 for i in range(len(nums))]
#     for i in range(len(nums)):
#         for j in range(i):
#             if(nums[i]>nums[j]):
#                 dp[i]=max(dp[i],dp[j]+1)
#     return dp
# print(max(fun([10,9,2,5,3,7,101,16,18])))


#全排列--动态规划版
# class Solution:
#     def permute(self, nums):
#         if not nums:
#             return None
#         dp = [[] for i in range(len(nums))]
#         dp[0] = [[nums[0]]]
#         for i in range(1, len(nums)):#1-4
#             p=dp[i-1]
#             for ans in dp[i-1]:
#                 a=ans
#                 b=nums[i]
#                 dp[i].append(ans+[nums[i]])
#                 for j in range(len(ans)):#换位置插入
#                     a=ans[:j]
#                     b=ans[j+1:]
#                     c=ans[j]
#                     dp[i].append(ans[:j]+[nums[i]]+ans[j+1:]+[ans[j]])
#         return dp[-1]
# ss=Solution()
# print(len(ss.permute([1,2,3,4,5])))


#516最长回文子序列
#初尝试时错误--111处运行时报错越界，range(1,1+len(nums))，不应该加1
#
# def fun(nums):
#     dp=[[[] for i in range(len(nums))]for i in range(len(nums))]
#     for i in range(len(nums)):
#         dp[i][i]=1
#         # for j in range(len(nums)):
#         for j in range(0, i):
#             # if(i>j):
#                 dp[i][j]=0
#     for j in range(1,len(nums)):#111
#         for i in range(j-1,-1,-1):
#             a=i
#             b=j
#             if(nums[i]==nums[j]):
#                 dp[i][j]=dp[i+1][j-1]+2
#             else:
#                 dp[i][j]=max(dp[i+1][j],dp[i][j-1])
#     return max(map(max,dp))
# # print fun("abcdb")
# print(fun("bbbab"))


#474 一和零--动态规划
"""思路：
    1.dp赋值本身，符合条件则为1，否则为0
    2.遍历字符串，对dp进行更新，若当前为第i个
        1)只有i自身dp不为0才进行
        2)找i-1中的所有与自己相加，符合的dp记录更新
    3。i=len(str)的记录
    中的dp最大值即为所求
"""

# def fun(strs,m,n):
#     def tong(s, k):
#         count = 0
#         for i in range(len(s)):
#             ss = s[i]
#             if (s[i] == k):
#                 count += 1
#         return count
#     dp=[[] for j in range(len(strs))]
#     for i in range(len(strs)):
#         zero = tong(strs[i], '0')
#         dp[i].append([1] + [zero] + [len(strs[i]) - zero])
#         ##111   Wrong！不能不加单引号，不然字符无法与数字相等，判断会出错
#         if dp[i][0][1]>m or dp[i][0][2]>n:
#             dp[i][0][0]=0
#         pp=dp[i-1]
#         for k in range(1,i+1):
#             for j in dp[i - k]:
#                 if dp[i][0][0] != 0 and (j[1] + dp[i][0][1]) <= m and (j[2] + dp[i][0][2]) <= n:
#                     dp[i].append([j[0] + dp[i][0][0]] + [j[1] + dp[i][0][1]] + [j[2] + dp[i][0][2]])
#
#     max=0
#     for j in range(1,len(strs)):
#         for i in dp[len(strs)-j]:#222   Wrong：只对最后一列求了最大值
#             if i[0]>max:
#                 max=i[0]
#     return max
"""   ↑↑↑↑  超时了  ↑↑↑  """

#改版
#
# def findMaxForm(strs, m, n) :
#     dp = [[[0] * (n + 1) for _ in range(m + 1)] for _ in range(len(strs) + 1)]
#     for i in range(1, len(strs) + 1):
#         ones = strs[i - 1].count("1")
#         zeros = strs[i - 1].count("0")
#         for j in range(m + 1):
#             for k in range(n + 1):
#                 dp[i][j][k] = dp[i - 1][j][k]
#                 if j >= zeros and k >= ones and dp[i][j][k] < dp[i - 1][j - zeros][k - ones] + 1:#求装与不装的最优
#                     dp[i][j][k] = dp[i - 1][j - zeros][k - ones] + 1
#     return dp[-1][-1][-1]
# strs = ["10","0001","111001","1","0"]
# m=5
# n=3
# print(findMaxForm(strs,m,n))



#背包问题的变体 --416 分割等和子集 （通过但有错
# def fun(strs):
#     n=sum(strs)
#     if(n%2!=0):
#         return False
#     else:
#         n/=2
#     dp=[[False for _ in range(n+1)] for j in range(len(strs))]#n+1:还有背包容量为0的时候
#     for i in range(len(strs)):
#         dp[i][0]=True  #Error!!第一个字符串判断时，dp[i][j]=dp[i-1][j]会出错，要最开始统一赋值
#         for j in range(1,n+1):
#             if(j-strs[i]<0):#装不下
#                 t=i-1
#                 dp[i][j]=dp[i-1][j]
#             else:
#                 dp[i][j]=dp[i-1][j] or dp[i-1][j-strs[i]]#Amazing
#     return dp[-1][-1]
# print(fun([1, 5, 11, 5]))


#背包问题的变体 --416
# def fun(strs):
#     n=sum(strs)
#     if(n%2!=0):
#         return False
#     else:
#         n/=2
#     dp=[[False for _ in range(n+1)] for j in range(len(strs))]#n+1:还有背包容量为0的时候
#     for i in range(len(strs)):
#         dp[i][0] = True
#     for i in range(len(strs)):
#         for j in range(1,n+1):
#             if(j-strs[i]<0):#装不下
#                 t=i-1
#                 dp[i][j]=dp[i-1][j]
#             else:
#                 dp[i][j]=dp[i-1][j] or dp[i-1][j-strs[i]]#Amazing
#     return dp[-1][-1]
# print(fun([1, 5, 11, 5]))


#股票的最大利润--dp---剑指 Offer 63
# def fun(nums):
#     if not nums:
#         return 0
#     dp= [0 for i in range(len(nums))]
#     for i in range(len(nums)):
#         for j in range(i):
#             if(nums[i]-nums[j]>0):
#                 dp[i]=max((nums[i]-nums[j]),dp[i])
#     return max(dp)
# print(fun([7,1,5,3,6,4]))

#  股票的最大利润--dp---剑指 Offer 63 ↑超时  ↓
# class Solution(object):
#     def maxProfit(self, prices):
#         before = 0
#         m=0
#         prices=[prices[i] - prices[i - 1]for i in range(1, len(prices))]# Amazing！达达可真厉害
#         for i in range(len(prices)):
#             before = max(before+prices[i], 0)
#             if before>m:
#                 m=before
#         return m
#         """
#         :type prices: List[int]
#         :rtype: int
#         """
# s=Solution()
# print(s.maxProfit([7,1,5,3,6,4]))


#1143 最长公共子序列
"""思路:
    1.找两个中较长的一个为背包，装另一个  --->转为背包问题 if a：abcde  b：acd
    2.背包容量：较长字符串的a的字母可视为背包容量
      物品数：b中的每一个字符为要放入背包的物品，即三个待存放的物品
      物品重量：对应背包容量，每个字符的重量为其字母本身
      价值：即为所求。可以存放的物品数量
    3.dp构造：二维数组。dp[i][j]即为当前背包容量为a[j]时，且当前有i个物品时，最大能存放的价值，即物品数量

    题目：https://leetcode-cn.com/problems/longest-common-subsequence/
"""
# class Solution(object):
#     def  longestCommonSubsequence(self, text1, text2):
#         if not text2 or not text1:
#             return 0
#         bao=text1
#         things=text2
#         if len(text1)<len(text2):
#             bao= text2
#             things=text1
#         dp= [[0 for _ in range(len(bao)+1)]for _ in range(len(things)+1)]
#
#         for i in range(1,1+len(things)):
#             for j in range(1,1+len(bao)):
#                 if(bao[j-1]==things[i-1]):
#
#                     dp[i][j]=dp[i-1][j-1]+1
#                 else:
#                     dp[i][j]=max(dp[i-1][j],dp[i][j-1])
#         return dp[-1][-1]
# s=Solution()
# print s.longestCommonSubsequence("abcde","cce")


#学长
'''
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
            
'''
# def fun(n,w,wt,val):
#     if not wt or not val:
#         return 0
#     dp=[[0 for _ in range(w+1)]for _ in range(n+1)]
#     for i in range(1,n+1):
#         for j in range(1,w+1):
#             dp[i][j]=dp[i-1][j]#注意=dp[i-1][j],而不是[i-1][j-1]
#             if(j>=wt[i-1]):
#                 dp[i][j] = max(dp[i][j], dp[i - 1][j - wt[i-1]] + val[i-1])
#     return dp[-1][-1]
# print fun(3,4,[2,1,3],[4,2,3])

#518零钱兑换II
# def fun(m,coins):
#     dp=[[0 for _ in range(m+1)] for _ in range(len(coins)+1)]
#     for i in range(len(coins)+1):
#         dp[i][0]=1
#     for i in range(1,len(coins)+1):
#         for j in range(1,m+1):
#             dp[i][j]=dp[i-1][j]
#             if(j>=coins[i-1]):
#                 dp[i][j]=dp[i-1][j]+dp[i][j-coins[i-1]]
#     return dp[-1][-1]
# print fun(5,[1,2,5])


#零钱兑换 322
# def coinChange(coins,amount):
#     dp = [[1000 for _ in range(amount + 1)] for _ in range(len(coins) + 1)]
#     for i in range(len(coins) + 1):
#         dp[i][0] = 0
#     for i in range(1,len(coins)+1):
#         for j in range(1,amount+1):
#             if(j>=coins[i-1]):
#                 dp[i][j]=min(dp[i-1][j],dp[i][j-coins[i-1]]+1)
#             else:
#                 dp[i][j]=dp[i-1][j]
#     if dp[-1][-1]==1000 :
#         return -1
#     else:
#         return dp[-1][-1]
# print coinChange([2,5,1],11)
# print coinChange([7,3,2],7)

#最小覆盖子串
#动态规划，超时了
# def fun(s,t):
#     res=""
#     def record(t):
#         num=len(t)
#         temp = [['' for _ in range(num)], [1 for _ in range(num)]]
#         j=0
#         for i in range(num):
#             if j==0 or t[i] not in temp[0]:
#                 temp[0][j]=t[i]
#                 for k in range(t.count(t[i])):
#                     temp[1][j]-=1
#             j+=1
#         return temp
#
#     temp = record(t)
#     temp1=temp[1][:]
#     for i in range(len(s)):
#         temp[1]=temp1[:]
#         if s[i] in temp[0]:
#             for j in range(i, -1, -1):
#                 if s[j] in t:
#                     temp[1][temp[0].index(s[j])] += 1
#                 if min(temp[1]) == 1:  # 每个字符都找到了
#                     if len(res) == 0 or len(res) > i - j + 1:  # 当前res较长
#                         res = s[j:i + 1]  # 左闭右开
#
#     if(min(temp[1])==0):
#         return ""#一次符合的也没有
#     return res
# print fun("ADOBECODEBANC", "ABC")
# print fun("aab", "aab")

#
# #滑窗解法
# def minWindow(s, t):
#     windows = dict()
#     needs = dict()
#     for i in t:
#         if i in needs:
#             needs[i] = needs[i] + 1
#         else:
#             needs[i] = 1
#     right = left = 0
#     void = 0
#     res = float("inf")
#     start = 0
#     while right < len(s):
#         c = s[right]
#         right += 1
#         if c in needs:
#             if c in windows:
#                 windows[c] = windows[c] + 1
#             else:
#                 windows[c] = 1
#             if windows[c] == needs[c]:
#                 void += 1
#             while void == len(needs):
#                 d = s[left]
#                 left += 1
#                 if right - left < res:
#                     start = left
#                     res = right - left
#                 if d in needs:
#                     if needs[d] == windows[d]:
#                         void -= 1
#                     windows[d] = windows[d] - 1
#     return s[start - 1:start + res] if res != float("inf") else ""
# print minWindow("ADOBECODEBANC", "ABC")
# # print fun("aab", "aab")

#try一下啊
# from collections import Counter
# def fun(s,t):
#     windows=dict()
#     needs=dict(Counter(t[:]))
#     right=left=0
#     volid=0
#     res=float("inf")
#     start=0
#     while right<len(s):
#         c=s[right]
#         right+=1
#         if c in needs:
#             if c in windows:
#                 windows[c]+=1
#             else:
#                 windows[c]=1
#             if windows[c]==needs[c]:
#                 volid+=1
#             while len(needs)==volid:
#                 d=s[left]
#                 left+=1
#                 if right-left+1<res:#这里不太一样
#                     start=left-1
#                     res=right-left+1
#                 if d in needs:
#                     if windows[d]==needs[d]:
#                         volid-=1
#                     windows[d]-=1
#     if res==float("inf"):
#         return ""
#     return s[start:res+start]
# # print fun("ab", "a")
# print fun("ADOBECODEBANC",  "ABC")
# # # print fun("aab", "aab")



#字符串的排列
# def checkInclusion(s1, s2) :
#     windows = dict()
#     needs = dict()
#     for i in s1:
#         if i in needs:
#             needs[i] = needs[i] + 1
#         else:
#             needs[i] = 1
#     right = left = 0
#     voiad = 0
#     while right < len(s2):
#         c = s2[right]
#         right += 1
#         if c in needs:
#             if c in windows:
#                 windows[c] = windows[c] + 1
#             else:
#                 windows[c] = 1
#             if windows[c] == needs[c]:
#                 voiad += 1
#             while voiad == len(needs):
#                 d = s2[left]
#                 if right - left == len(s1):
#                     return True
#                 if d in needs:
#                     if windows[d] == needs[d]:
#                         voiad -= 1
#                     windows[d] = windows[d] - 1
#                 left = left + 1
#     return False

#自己练一遍
# def checkInclusion(p,s):
#     windows=dict()
#     needs=dict()#忘记加括号了
#     for i in p:
#         if i in needs:
#             needs[i]+=1
#         else:
#             needs[i]=1
#
#     right=left=0
#     valid=0
#     while right<len(s):
#         c=s[right]
#         right+=1
#         if c in needs:
#             if c in windows:
#                 windows[c]+=1
#             else:
#                 windows[c]=1
#             if windows[c]==needs[c]:
#                 valid+=1
#             while valid==len(needs):
#                 d=s[left]
#                 left+=1
#                 if right- left+1==len(p):
#                     return True
#                 if d in needs:
#                     if windows[d]==needs[d]:
#                         valid-=1
#                     windows[d]-=1
#     return False
# print checkInclusion("ab", "eidboaoo")

#438找到字符串中所有字母异位词
#一遍过，嘻嘻
# from collections import  Counter
# def fun(s,p):
#     windows=dict()
#     needs = dict(Counter(p[:]))
#     right=left=0
#     voild=0
#     res=[]
#     while right<len(s):
#         c=s[right]
#         right+=1
#         if c in needs:
#             if c in windows:
#                 windows[c]+=1
#             else:
#                 windows[c]=1
#             if windows[c]==needs[c]:
#                 voild+=1
#             while voild==len(needs):
#                 d=s[left]
#                 left+=1
#                 if right-left+1==len(p):
#                     res.append(left-1)
#                 if d in needs:
#                     if windows[d]==needs[d]:
#                         voild-=1
#                     windows[d]-=1
#     return res
# print fun("cbaebabacd" ,"abc")

#3 无重复字符的最长子串
# def lengthOfLongestSubstring(s):
#     windows=dict()
#     right=left=0
#     res=0
#     while right<len(s):
#         c=s[right]
#         right+=1
#         if c in windows :
#             windows[c]+=1
#         else:
#             windows[c]=1
#         while windows[c]>1:
#             d=s[left]
#             left+=1
#             windows[d]-=1
#         res=max(res,right-left)
#     return res


#前k个高频元素
from collections import Counter
# def topKFrequent( nums, k):
#     nlist = dict(Counter(nums[:]))
#     resN = []
#     finanal = []
#     for i in nlist:
#         resN.append(nlist[i])
#     j=0
#     for i in nlist:
#         if resN[j] ==max(resN):
#             resN[j]=0
#             finanal.append(i)
#         j+=1
#         if len(finanal)==k:
#             return finanal
#     return finanal
# def topKFrequent(nums, k) :
#     nlist = dict(Counter(nums[:]))
#     resN = []
#     finanal = []
#     for i in nlist:
#         resN.append(nlist[i])
#     for i in nlist:
#         max1=max(resN)
#         j=0
#         for l in nlist:
#             if resN[j] == max1:
#                 resN[j] = 0
#                 finanal.append(l)
#                 # j += 1
#             if len(finanal) == k:
#                 return finanal
#             j+=1
#     return finanal
# print (topKFrequent([3,2,3,1,2,4,5,5,6,7,7,8,2,3,1,1,1,10,11,5,6,2,4,7,8,5,6],10))



#198打家劫舍

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






