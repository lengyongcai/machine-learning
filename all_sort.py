#coding:utf-8


lists = [4, 5, 1, 6, 9, 3, 10, 2]


# 1 插入排序

# 插入排序的核心思想：将数组中的所有元素依次跟前面已经排好的元素进行比较，
# 如果选择的元素比已排序的元素小，则交换，直到全部元素都比较过。

# 插入排序算法实现
def insert_sort(lists):
    for x in range(1, len(lists)):
        for i in range(x - 1, -1, -1):
            if lists[i] > lists[i+1]:
                temp = lists[i+1]
                lists[i+1] = lists[i]
                lists[i] = temp

    return lists

# insert_sort = insert_sort(lists)
# print("insert_sort:", insert_sort)


# 2 冒泡排序

# 冒泡排序的核心思想：将序列中的左右元素依次比较，保证右边的元素始终大于左边的元素。
# 第一轮结束后，序列最后一个元素一定是当前序列的最大值，在序列中剩下的n-1个元素再次执行上面的步骤。

# 冒泡排序算法实现
def bubble_sort(lists):

    for x in range(len(lists)):
        for i in range(x + 1, len(lists)):
            if lists[x] > lists[i]:
                # lists[x], lists[i] = lists[i], lists[x]
                temp = lists[i]
                lists[i] = lists[x]
                lists[x] = temp
    return lists


# bubble_sort = bubble_sort(lists)
# print("bubble_sort:", bubble_sort)


# 3 希尔排序

# 希尔排序的核心思想：将待排序数组按照步长gap 进行分组，
# 然后将每组的元素利用直接插入排序的方法进行排序，
# 每次将 gap 折半减小，循环上述操作，当 gap=1 时，利用直接插入完成排序。

# 希尔排序算法实现
def shell_sort(lists):
    gap = int(len(lists)/2)

    while gap >= 1:
        for x in range(gap, len(lists)):
            for i in range(x - gap, -1, -gap):
                if lists[i] > lists[i + gap]:
                    temp = lists[i + gap]
                    lists[i + gap] = lists[i]
                    lists[i] = temp

        gap = int(gap/2)

    return lists


# shell_sort = shell_sort(lists)
# print("shell_sort:", shell_sort)


# 4 选择排序

# 选择排序核心思想：第一趟在待排序记录 r1,r2,...,rn 中选出最小的元素，将于 r1交换，
# 第二趟在待排序记录 r2,r3,...rn中选出最小的元素，将与 r2交换，
# 以此类推，第i趟在待排序记录 ri,...rn中选出最小的元素，将与 ri交换，,
# 使有序序列不断增长直到全部排序完毕。

# 选择排序算法实现
def select_sort(lists):
    for x in range(len(lists)):
        minimum = lists[x]
        for i in range(x+1, len(lists)):
            if minimum > lists[i]:
                temp = lists[i]
                lists[i] = minimum
                minimum = temp

        lists[x] = minimum

    return lists


# select_sort = select_sort(lists)
# print("select_sort:", select_sort)


# 5 快速排序

# 快速排序的核心思想：将排序列表list的第一个作为left和key 最后一个作为right，
# 通过一趟排序将要排序的数据列表以key 分割成独立的部分，其中前面一部分数据都小于后面一部分数据。
# 然后按照此方法对这两部分数据分别进行快速排序，整个过程可以递归进行。
# 以此达到整个数据变成有序序列。

# 快速排序算法实现
def quick_sort(lists, _left, _right):
    if _left >= _right:
        return lists

    left = _left
    right = _right
    key = lists[left]  # 待排序的第一个元素作为基准元素

    while left != right:  # //从左右两边交替扫描，直到left = right

        while right > left and lists[right] >= key:
            right -= 1              # 从右往左扫描，找到第一个比基准元素小的元素
        lists[left] = lists[right]  # 找到这种元素arr[right]后与arr[left]交换

        while right > left and lists[left] <= key:
            left += 1               # 从左往右扫描，找到第一个比基准元素大的元素
        lists[right] = lists[left]  # 找到这种元素arr[left]后，与arr[right]交换

    lists[right] = key   # 基准元素归位

    quick_sort(lists, _left, left - 1)    # 对基准元素左边的元素进行递归排序
    quick_sort(lists, right + 1, _right)  # 对基准元素右边的进行递归排序

    return lists


# quick_sort = quick_sort(lists, 0, len(lists)-1)
# print("quick_sort:", quick_sort)


# 6 堆排序

# 堆排序核心思想：将无序序列建成一个堆，得到关键字最小(最大)记录，输出堆顶的最小(最大)值后，
# 使剩余的 n-1 个元素又建成一个堆，则得到 n 个元素的次小值，重复执行，得到一个有序序列。
# 输出堆顶元素之后，以堆中最小的元素替代之，然后将根据节点值与左右子树的根节点值进行比较
# 并与其中小者进行交换，重复上述操作，直到叶子结点，得到新的堆。

# 堆排序算法实现
def adjust_heap(lists, i, size):  # 调整堆

    lchild = 2 * i       # i的左孩子节点序号
    rchild = 2 * i + 1   # i的右孩子节点序号
    max = i              # 临时变量

    if i < size / 2:     # 如果i不是叶节点就不用进行调整
        if lchild < size and lists[lchild] > lists[max]:
            max = lchild
        if rchild < size and lists[rchild] > lists[max]:
            max = rchild
        if max != i:
            lists[max], lists[i] = lists[i], lists[max]

            adjust_heap(lists, max, size)  # 避免调整之后以max为父节点的子树不是堆

def build_heap(lists, size):          # 建立堆
    for i in range(0, size/2)[::-1]:  # 非叶节点最大序号值为size/2
        adjust_heap(lists, i, size)


def heap_sort(lists):     # 堆排序
    size = len(lists)
    build_heap(lists, size)

    for i in range(0, size)[::-1]:
        lists[0], lists[i] = lists[i], lists[0]   # 交换堆顶和最后一个元素，即每次将剩余元素中的最大者放到最后面

        adjust_heap(lists, 0, i)  # 重新调整堆顶节点成为大顶堆

    return lists


# heap_sort = heap_sort(lists)
# print("heap_sort:", heap_sort)


# 7 归并排序


def merge(left, right):
    i, j = 0, 0
    result = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:]
    result += right[j:]
    return result


def merge_sort(lists):
    # 归并排序
    if len(lists) <= 1:
        return lists
    num = len(lists) / 2
    left = merge_sort(lists[:num])
    right = merge_sort(lists[num:])
    return merge(left, right)

merge_sort = merge_sort(lists)
print("merge_sort:", merge_sort)


