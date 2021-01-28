#55跳跃游戏
class Solution:
    def canJump(self, nums) -> bool:
        dp=[[] for i in range(len(nums)+1)]
        dp[0]=0
        for i in range(1,len(nums)):
            dp[i]=max(dp[i-1]-1,nums[i-1])
            if dp[i]==0:
                return False
        return True
#75颜色分类
class Solution:
    def sortColors(self, nums) -> None:
        p1 = 0
        cur=0
        p2=len(nums)-1
        while cur<=p2:
            while cur<=p2 and nums[cur]==2:
                nums[cur],nums[p2]=nums[p2],nums[cur]
                p2-=1
            if nums[cur]==0:
                nums[p1],nums[cur]=nums[cur],nums[p1]
                p1+=1
            cur+=1
#48旋转图像
class Solution:
    def rotate(self, matrix) -> None:#matrix:: List[List[int]]
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

#打家劫舍
class Solution(object):
    def rob(self, nums):
        if not nums:
            return 0
        # if len(nums) == 1:
        #     return nums[0]
        dp = [[] for i in range(len(nums) + 1)]
        dp[0] = 0
        dp[1] = nums[0]
        for i in range(2, len(nums) + 1):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
        return dp[-1]

#102 二叉树的层序遍历
import Tree
def levelOrder( root) :
    if not root:return []
    queue=[]
    queue.append(root)
    res=[]
    while len(queue)>0:
        temp=[]#记录每一层的节点
        for i in range(len(queue)):
            node=queue.pop(0)
            temp.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        res.append(temp[:])#将每一层加入结果中
    return res
tree=Tree.TreeNodeTools()
root=tree.createTreeByrow([3,9,20,3,'null',15,7],0)
print(levelOrder(root))


#打家劫舍III
class Solution:
    def rob(self, root) -> int:
        def _rob(root):
            if not root: return 0, 0  # 偷，不偷

            left = _rob(root.left)
            right = _rob(root.right)
        # 偷当前节点, 则左右子树都不能偷
            v1 = root.val + left[1] + right[1]
        # 不偷当前节点, 则取左右子树中最大的值
            v2 = max(left) + max(right)
            return v1, v2
        return max(_rob(root))

















