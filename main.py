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

# def rob( nums):
#     if not nums:
#         return 0
#     dp=[[] for i in range(len(nums)+1)]
#     dp[0]=0
#     dp[1]=nums[0]
#     for i in range(2,len(nums)+1):
#         dp[i]=max(dp[i-1],dp[i-2]+nums[i-1])
#     return dp[-1]
# print(rob([1,2,3,1]))


# 达达：72 编辑距离
# def minDistance( word1, word2) :
#     dp=[[[] for i in range(len(word1)+1)] for _ in range(len(word2)+1)]
#
#     for i in range(len(word1)+1):
#         dp[0][i]=i
#     for i in range(len(word2)+1):
#         dp[i][0]=i
#
#     for i in range(1,len(word2)+1):
#         for j in range(1,len(word1)+1):
#             dp[i][j]=dp[i-1][j-1]
#             if word1[j-1]!=word2[i-1]:
#                 dp[i][j]=min(dp[i-1][j]+1,dp[i][j]+1,dp[i][j-1]+1)
#     return dp[-1][-1]
#
# print(minDistance("b",  ""))


#337 打家劫舍III
# def rob(nums):
#     dp=[[]*len(nums)]
#     ceng=1
#     for i in range(len(nums)):
#         for j in range
#
#     print(dp)
# print(rob([]))

#
# def rob(root):
#     def _rob(root):
#         if not root: return 0, 0  # 偷，不偷
#
#         left = _rob(root.left)
#         right = _rob(root.right)
#         # 偷当前节点, 则左右子树都不能偷
#         v1 = root.val + left[1] + right[1]
#         # 不偷当前节点, 则取左右子树中最大的值
#         v2 = max(left) + max(right)
#         return v1, v2
#
#     return max(_rob(root))
# print(rob([3,2,3,None,3,None,1]))

#287 寻找重复数
# def findDuplicate( nums):
#     for i in range(1, len(nums)):
#         for j in range(i - 1, -1, -1):
#             if nums[j] == nums[i]:
#                 return nums[i]
# print(findDuplicate([3,1,3,4,2]))


# def findDuplicate( nums):
#     size = len(nums)
#     left = 1
#     right = size - 1
#     while left < right:
#         mid = left + (right - left) // 2
#         cnt = 0
#         for num in nums:
#             if num <= mid:
#                 cnt += 1
#         if cnt > mid:  # 根据抽屉原理，小于等于 4 的数的个数如果严格大于 4 个
#                         # 此时重复元素一定出现在 [1, 4] 区间里
#             right = mid # 重复的元素一定出现在 [left, mid] 区间里
#         else:
#             left = mid + 1 # [mid + 1, right]
#     return left
# print(findDuplicate([3,1,3,4,2]))


#48旋转图像
# def rotate( matrix) :
#     pos1, pos2 = 0, (len(matrix[0]) - 1)
#     while pos1 < pos2:
#         add = 0
#         while add < pos2 - pos1:
#             temp = matrix[pos1][pos1 + add]
#             matrix[pos1][pos1 + add] = matrix[pos2 - add][pos1]
#             matrix[pos2 - add][pos1] = matrix[pos2][pos2 - add]
#             matrix[pos2][pos2 - add] = matrix[pos1 + add][pos2]
#             matrix[pos1 + add][pos2] = temp
#             add += 1
#         pos1+=1
#         pos2-=1


#75颜色分类
# def sortColors(nums) :

#     p0 = cur = 0;p2 = len(nums) - 1
#     while cur <= p2:
#         # 当nums[cur]=2时
#         while cur <= p2 and nums[cur] == 2:
#             nums[cur], nums[p2] = nums[p2], nums[cur]
#             p2 -= 1
#         # 当nums[cur]=0时
#         if nums[cur] == 0:
#             nums[cur], nums[p0] = nums[p0], nums[cur]
#             p0 += 1
#         cur += 1
#     return nums
# # print(sortColors([2,0,2,1,1,0]))
# print(sortColors([1,1,2,2,2,1,0,0]))

#打家劫舍III
# import Tree
# def rob(root) -> int:
#     def _rob(root):
#         if not root: return 0, 0  # 偷，不偷
#         left = _rob(root.left)
#         right = _rob(root.right)
#         # 偷当前节点, 则左右子树都不能偷
#         v1 = root.val + left[1] + right[1]#记录偷该节点的钱数
#         # 不偷当前节点, 则取左右子树中最大的值
#         v2 = max(left) + max(right)#记录不偷当前节点的钱数，那下一层一定偷
#                                     # 将左孩子的最大偷钱数加上右孩子的最大偷钱数
#         return v1, v2
#
#     return max(_rob(root))
# tree=Tree.TreeNodeTools()
# root=tree.createTreeByrow([3,4,5,1,3,'null',1],0)
# print(rob(root))

#102 二叉树的层序遍历
# import Tree
# def levelOrder( root) :
#     if not root:return []
#     queue=[]
#     queue.append(root)
#     res=[]
#     while len(queue)>0:
#         temp=[]#记录每一层的节点
#         for i in range(len(queue)):
#             node=queue.pop(0)
#             temp.append(node.val)
#             if node.left:
#                 queue.append(node.left)
#             if node.right:
#                 queue.append(node.right)
#         res.append(temp[:])#将每一层加入结果中
#     return res
# tree=Tree.TreeNodeTools()
# root=tree.createTreeByrow([3,9,20,3,'null',15,7],0)
# print(levelOrder(root))


#反转链表
# import ListNode
# def reverseList(head) :
#     if not head:
#         return None
#     pre, cur = head, head.next
#     pre.next = None
#     while cur:
#         temp = cur.next
#         cur.next = pre
#         pre, cur = cur, temp
#     return pre
# listnode=ListNode
# ln=listnode.list_2_linknode([1,2,3,4,5])
# node=reverseList(ln)
# print(listnode.travel_list(node))

#215 第k个最大元素
#快排：超时
# class Solution:
#     def findKthLargest(self, nums: List[int], k: int) -> int:
#         def partition(arr, left, right):
#             povit = arr[right]
#             i = left - 1
#             for j in range(left, right):
#                 if povit < arr[j]:
#                     i += 1  # 比povit小的值的个数
#                     arr[i], arr[j] = arr[j], arr[i]
#             arr[i + 1], arr[right] = arr[right], arr[i + 1]
#             return i + 1
#         def quicksort(arr, left, right):
#             if left < right:
#                 q = partition(arr, left, right)  # 本次被确定数字的位置
#                 quicksort(arr, q + 1, right)
#                 quicksort(arr, left, q - 1)
#             return arr
#         list1=quicksort(nums,0,len(nums)-1)
#         return list1[k-1]


#快排
# def partition(arr,left,right):
#     povit=arr[right]
#     i=left-1
#     for j in range(left,right):
#         if povit>arr[j]:
#             i+=1#比povit小的值的个数
#             arr[i],arr[j]=arr[j],arr[i]
#     arr[i+1],arr[right]=arr[right],arr[i+1]
#     return i+1
# def quicksort(arr,left,right):
#     if left<right:
#         q=partition(arr,left,right)#本次被确定数字的位置
#         quicksort(arr,q+1,right)
#         quicksort(arr,left,q-1)
#     return arr
#
#
# arr=[6, 12, 27, 34, 21, 4, 9, 8, 11, 54, 3, 7, 39]
# print(quicksort(arr,0,len(arr)-1))


#堆排
# def HeapAdjust(lst,k,n):
#     while(2*k+1<n):
#         j=2*k+1
#         if j+1<n and lst[j]>lst[j+1]:
#             j=j+1#j=小的那一个
#         if lst[j]<lst[k]:
#             temp=lst[k]
#             lst[k]=lst[j]
#             lst[j]=temp#最小的放在根节点
#             k=j#要调整的节点
#         else:
#             break
#     return lst
# def HeapSort(lst):
#     n=len(lst)
#     for i in range(int(n/2)-1,-1,-1):
#         lst=HeapAdjust(lst,i,n)
#     for i in range(n-1,0,-1):
#         lst[0],lst[i]=lst[i],lst[0]
#         lst=HeapAdjust(lst,0,i)
#     return lst
# a=[1,5,2,8,3,4,6,9,7]
# result=HeapSort(a)
# print(result)




#221最大正方形
# def maximalSquare(matrix) -> int:
#     # dp = [[[]* (len(matrix)+1)] * (len(matrix)+1)]
#     dp = [[0 for i in range(len(matrix[0]))] for i in range(len(matrix))]
#     for i in range(len(dp)):
#         dp[i][0] = int(matrix[i][0])
#     for i in range(len(dp[0])):
#         dp[0][i] = int(matrix[0][i])
#     for i in range(1, len(dp)):
#         for j in range(1, len(dp[0])):
#             dp[i][j] = (int)(matrix[i][j])
#             if (matrix[i][j] == '1' and matrix[i][j - 1] == '1' and matrix[i - 1][j] == '1' and matrix[i - 1][j - 1] == '1'):
#                 dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
#     max1=max(map(max, dp))
#     return max1*max1
# print(maximalSquare([["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]))
#


#85最大矩形
# def maximalRectangle(matrix) -> int:
#     if not matrix:
#         return 0
#     dp = [[[] for i in range(len(matrix[0]))] for i in range(len(matrix))]
#     maxS = 0
#     maxL = 0
#     for i in range(len(dp)):
#         l = 0
#         # dp[i][0].append(int(matrix[i][0]))
#         # dp[i][0].append(int(matrix[i][0]))
#         for j in range(len(dp[0])):
#             if matrix[i][j] == '1':
#                 l += 1
#                 init = [1,1,1,1,1,1]
#             else:
#                 l = 0
#                 init = [0,0,0,0,0,0]
#             dp[i][j]=init
#         if l > maxL: maxL = l
#     for j in range(len(dp[0])):
#         s = 0
#         for i in range(len(dp)):
#             if matrix[i][j] == '1':
#                 s += 1
#             else:
#                 s = 0
#         if s > maxS: maxS = s
#     maxQ = 0
#     for i in range(1, len(matrix)):
#         for j in range(1, len(matrix[0])):
#             if matrix[i][j] == '1' and matrix[i - 1][j - 1] == '1' and matrix[i][j - 1] == '1' and matrix[i - 1][
#                 j] == '1':
#                 v0 = dp[i - 1][j - 1][0]+1
#                 v1 = dp[i - 1][j - 1][1]+1
#                 v2 = dp[i - 1][j][2]
#                 v3 = dp[i - 1][j][3] +1
#                 v4 = dp[i][j - 1][4] +1
#                 v5 = dp[i][j-1][5]
#                 dp[i].pop(j)
#                 if max(v0 * v1 ,v2 * v3 ,v4 * v5 )> maxQ:
#                     maxQ = max(v0 * v1 ,v2 * v3 ,v4 * v5 )
#                 dp[i].insert(j,[v0,v1,v2,v3,v4,v5])
#     return max(maxQ, maxL, maxS)
# print(maximalRectangle([["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]))


#盛最多水的容器
# def maxArea(height):
#     i, j, res = 0, len(height) - 1, 0
#     while i < j:
#         if height[i] < height[j]:
#             res = max(res, height[i] * (j - i))
#             i += 1
#         else:
#             res = max(res, height[j] * (j - i))
#             j -= 1
#     return res
# print(maxArea([1,8,6,2,5,4,8,3,7]))


#240 搜索二位矩阵
"""
搜索空间的缩减
我们可以将已排序的二维矩阵划分为四个子矩阵，其中两个可能包含目标，其中两个肯定不包含。

算法：
由于该算法是递归操作的，因此可以通过它的基本情况和递归情况的正确性来判断它的正确性。

基本情况 ：
对于已排序的二维数组，有两种方法可以确定一个任意元素目标是否可以用常数时间判断。
第一，如果数组的区域为零，则它不包含元素，因此不能包含目标。其次，如果目标小于数
组的最小值或大于数组的最大值，那么矩阵肯定不包含目标值。

递归情况：
如果目标值包含在数组内，因此我们沿着索引行的矩阵中间列 ，
matrix[row-1][mid] < target < matrix[row][mid]

（很明显，如果我们找到 target ，我们立即返回 true）。
现有的矩阵可以围绕这个索引分为四个子矩阵；左上和右下子矩阵不能包含目标
（通过基本情况部分来判断），所以我们可以从搜索空间中删除它们 。
另外，左下角和右上角的子矩阵是二维矩阵，因此我们可以递归地将此算法应用于它们。





"""
# class Solution:
#     def searchMatrix( matrix, target):
#         # an empty matrix obviously does not contain `target`
#         if not matrix:
#             return False
#         def search_rec(left, up, right, down):
#             # this submatrix has no height or no width.
#             if left > right or up > down:
#                 return False
#             # `target` is already larger than the largest element or smaller
#             # than the smallest element in this submatrix.
#             elif target < matrix[up][left] or target > matrix[down][right]:
#                 return False
#
#             mid = left + (right - left) // 2
#
#             # Locate `row` such that matrix[row-1][mid] < target < matrix[row][mid]
#             row = up
#             while row <= down and matrix[row][mid] <= target:
#                 if matrix[row][mid] == target:
#                     return True
#                 row += 1
#
#             return search_rec(left, row, mid - 1, down) or search_rec(mid + 1, up, right, row - 1)
#
#         return search_rec(0, 0, len(matrix[0]) - 1, len(matrix) - 1)


# def searchMatrix(matrix, target):
#     if not matrix:
#         return False
#     def search_rec(left, up, right, down):
#         if left > right or up > down:
#             return False
#         # 如果target比矩阵最小值还小，或比最大值还大
#         elif target < matrix[up][left] or target > matrix[down][right]:
#             return False
#         mid = left + (right - left) // 2
#
#         row = up
#         while row <= down and matrix[row][mid] <= target:
#             if matrix[row][mid] == target:
#                 return True
#             row += 1
#         return search_rec(left, row, mid - 1, down) or search_rec(mid + 1, up, right, row - 1)
#     return search_rec(0, 0, len(matrix[0]) - 1, len(matrix) - 1)
# print(searchMatrix([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]],5))



#二分
"""
首先，我们确保矩阵不为空。那么，如果我们迭代矩阵对角线，从当前元素对列和行搜索，
我们可以保持从当前 (row,col)(row,col) 对开始的行和列为已排序。 因此，
我们总是可以二分搜索这些行和列切片。我们以如下逻辑的方式进行 : 在对角线上迭
代，二分搜索行和列，直到对角线的迭代元素用完为止（意味着我们可以返回 false ）
或者找到目标（意味着我们可以返回 true ）。binary search 函数的工作原理和普
通的二分搜索一样,但需要同时搜索二维数组的行和列。



"""
# class Solution:
#     def binary_search(self, matrix, target, start, vertical):
#         lo = start
#         hi = len(matrix[0]) - 1 if vertical else len(matrix) - 1
#         while hi >= lo:
#             mid = (lo + hi) // 2
#             if vertical:  # searching a column
#                 if matrix[start][mid] < target:
#                     lo = mid + 1
#                 elif matrix[start][mid] > target:
#                     hi = mid - 1
#                 else:
#                     return True
#             else:  # searching a row
#                 if matrix[mid][start] < target:
#                     lo = mid + 1
#                 elif matrix[mid][start] > target:
#                     hi = mid - 1
#                 else:
#                     return True
#         return False
#     def searchMatrix(self, matrix, target):
#         if not matrix:
#             return False

#         for i in range(min(len(matrix), len(matrix[0]))):
#             vertical_found = self.binary_search(matrix, target, i, True)
#             horizontal_found = self.binary_search(matrix, target, i, False)
#             if vertical_found or horizontal_found:
#                 return True
#         return False


#84 柱状图中最大的矩形
# def largestRectangleArea( heights):
#     left, right, res = 0, len(heights) - 1, 0
#     while left <= right:
#         minH = heights[left]
#         if heights[left] <= heights[right]:
#             for i in range(left, right + 1):
#                 if heights[i] < minH: minH = heights[i]
#             res = max(res, minH * (right+1 - left))
#             left += 1
#         else:
#             for i in range(left, right + 1):
#                 if heights[i] < minH: minH = heights[i]
#             res = max(res, minH * (right+1 - left))
#             right -= 1
#     return res
# print(largestRectangleArea([4,2,0,3,2,4,3,3]))


#完全平方数
# def numSquares( n: int) -> int:
#     dp = [[] for _ in range(n + 1)]
#     dp[0] = 0
#     for i in range(1,n + 1):
#         dp[i] = i
#         for j in range(1,n):
#             t = j * j
#             if (t <= i):  # 小于等于！！
#                 b=dp[i - t] + 1
#                 dp[i] = min(dp[i - t] + 1, dp[i])
#             else:
#                 break
#     return dp[-1]
# print(numSquares(12))


#矩形最大面积
# def largestRectangleArea(heights) -> int:
#     stack = []
#     heights = [0] + heights + [0]
#     res = 0
#     for i in range(len(heights)):
#         # print(stack)
#         while stack and heights[stack[-1]] > heights[i]:
#             tmp = stack.pop()
#             res = max(res, (i - stack[-1] - 1) * heights[tmp])
#         stack.append(i)
#     return res
#
# print(largestRectangleArea([2,1,5,6,2,3]))


#85最大矩形
# def maximalRectangle(matrix) -> int:
#     if not matrix: return 0
#     h, w = len(matrix), len(matrix[0])
#     # 记录当前位置上方连续“1”的个数
#     heights = [0] * (w+2)
#     res = 0
#     for i in range(h):
#         for j in range(1,w+1):
#             # 前缀和
#             heights[j] = heights[j] + 1 if matrix[i][j-1] == "1" else 0
#         # 单调栈
#         stack = []
#         # heights = [0] + heights + [0]
#         for k in range(len(heights)):
#             while stack and heights[stack[-1]] > heights[k]:
#                 tmp = stack.pop()
#                 res = max(res, heights[tmp] * (k - stack[-1] - 1))
#             stack.append(k)
#
#     return res
# print(maximalRectangle([["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]))


# def combinationSum( candidates, target) :
#     dp = [[[] for _ in range(target + 1)] for _ in range(len(candidates) + 1)]
#     # for i in range(len(dp[0])):
#     #     dp[0][i]='null'   #不需要初始化
#     for i in range(1, len(dp) ):
#         q=candidates[i-1]
#         for j in range(1, len(dp[0]) ):
#             dp[i][j] = dp[i - 1][j][:]
#             if j==q:
#                 dp[i][j].append([q])
#             if j > q and dp[i][j - q] != []:
#                     for k in dp[i][j - q]:
#                         dp[i][j].append(k+[q])
#     print(dp)
#     return dp[-1][-1]
# print(combinationSum([2,3,6,7],  7))


#比特位计数
# def countBits(num) :
#     if num == 0 or num == 1:
#         return num
#     dp = [[] for i in range(num + 1)]
#     dp[0] = 0
#     dp[1] = 1
#     p = 2
#     for i in range(2, num + 1):
#         if i == p:
#             dp[i] = 1
#             p *= 2
#             continue
#         t=i/2
#         if i%2 == 0:
#             dp[i] = dp[i - 1]
#         else:
#             dp[i] = dp[i - 1] + 1
#     return dp
# print(countBits(8))


# def maxProduct(nums) -> int:
#     if len(nums) == 1:
#         return nums[0]
#     iimax = nums[0]
#     iimin = nums[0]
#     imax = nums[0]
#     for i in range(1, len(nums)):
#         iimax = max(iimax * nums[i], nums[i], iimin * nums[i])
#         iimin = min(iimin * nums[i], nums[i], iimax * nums[i])
#
#         imax = max(iimax, imax)
#     return imax
# print(maxProduct([-4,-3,-2]))

#128
# def longestConsecutive( nums) :
#     dic = dict(Counter(nums[:]))
#     imin=min(dic)
#     del dic[imin]
#     count = 1
#     c=count
#     while dic:
#         if dic.__contains__(imin+1) :#dic[n2 + 1]:
#             count += 1
#             del dic[imin+1]
#             imin += 1
#         else:
#             imin = min(dic)
#             del dic[imin]
#             count=1
#         c = max(c, count)
#     return c
# print(longestConsecutive([6,7,9,10,2,3,4]))

#506
# def subarraySum( nums, k) -> int:
#     dp = [[0 for i in range(k + 1)] for i in range(len(nums) + 1)]
#     for j in range(1, k+1):
#         for i in range(1, len(nums)+1):
#             if nums[i-1] == j:
#                 dp[i][j] = dp[i - 1][j] + 1
#             elif nums[i-1] < j:
#                 dp[i][j] = dp[i - 1][j - 1]
#             else:
#                 dp[i][j] = dp[i - 1][j]
#     return dp[-1][-1]
# print(subarraySum([1,1,2,2,3,3,6],2))

#目标和
def findTargetSumWays( nums, S) -> int:
    sum = 0
    for i in nums:
        sum += i
    if sum < S:
        return 0
    dp = [[0 for i in range(sum * 2 + 1)] for i in range(len(nums) + 1)]
    dp[0][sum] = 1
    j = 0
    for i in range(1, len(nums) + 1):
        j += nums[i - 1]
        dp[i][sum-j]=dp[i][sum+j]=1
        for k in range(sum - j, sum + 1):
            if k - nums[i - 1] >= 0 and k + nums[i - 1] <= sum*2:
                dp[i][k] = dp[i][sum * 2 - k] = max(dp[i][k], dp[i - 1][k - nums[i - 1]] + dp[i - 1][k + nums[i - 1]])
            elif k - nums[i - 1] < 0 and k + nums[i - 1] <= sum*2:
                dp[i][k] = dp[i][sum * 2 - k] = max(dp[i][k], dp[i - 1][k + nums[i - 1]])
            elif k - nums[i - 1] >= 0 and k + nums[i - 1] > sum*2:
                dp[i][k] = dp[i][sum * 2 - k] = max(dp[i][k], dp[i - 1][k - nums[i - 1]])
    return dp[-1][sum + S]
print(findTargetSumWays([0,0,0,0,0,0,0,0,1], 1))

#优化
def findTargetSumWays(self, nums, S) -> int:
    sum = 0
    for i in nums:
        sum += i
    if sum < S:
        return 0
    dp = [[0 for i in range(sum * 2 + 1)] for i in range(len(nums) + 1)]
    dp[0][sum] = 1
    j = 0
    for i in range(1, len(nums) + 1):
        j += nums[i - 1]
        dp[i][sum - j] = dp[i][sum + j] = 1
        for k in range(sum - j, sum + 1):
            l = 0 if k - nums[i - 1] < 0 else k - nums[i - 1]
            r = 0 if k + nums[i - 1] > sum * 2 else k + nums[i - 1]
            dp[i][k] = dp[i][sum * 2 - k] = max(dp[i][k], dp[i - 1][l] + dp[i - 1][r])
    return dp[-1][sum + S]


# from collections import Counter
# #395 至少有 K 个重复字符的最长子串
# def longestSubstring(s: str, k: int) -> int:
#     c=dict(Counter(s))
#     res = {}
#     for i in s:
#         if i not in res:
#             res[i] = 1
#         else:
#             res[i] += 1
#     a=min(res)
#     a=min(c)
#     return 0
# print(longestSubstring("aaabb",1))










