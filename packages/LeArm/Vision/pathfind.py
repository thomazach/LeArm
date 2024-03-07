import cv2
import numpy as np
import Vision.learmVisionV2 as pp
import heapq

def create_graph(nodes, edges):
    graph = {node: [] for node in nodes}
    for edge in edges:
        start, end, distance = edge
        graph[start].append((end, distance))
        graph[end].append((start, distance))  # Assuming undirected graph
    return graph

def dijkstra(graph, start, end):
    queue = [(0, start, [])]  # Priority queue of (distance, node, path)
    seen = set()
    
    while queue:
        (distance, currNode, path) = heapq.heappop(queue)
        if currNode in seen:
            continue
        seen.add(currNode)
        
        # Return path if end node is reached
        if currNode == end:
            return path + [currNode]
        
        # Add neighbors to queue
        for (neighbor, weight) in graph[currNode]:
            if neighbor not in seen:
                heapq.heappush(queue, (distance + weight, neighbor, path + [currNode]))
    
    return None  # Path not found

def pain_and_anguish(pos, path):
    """die"""
    x1, y1 = pos
    closeNode = None
    closeDis = 999999999999999999999999999999999999999999999999999999999
    for i in range(0, len(path)):
        x2, y2 = path[i]
        dis = ((x1 - x2)**2 + (y1-y2)**2) ** 0.5
        if dis < closeDis:
            closeDis = dis
            closeNode = i
    
    try:
        nextNode = path[closeNode + 1]
    except:
        return 69420
    
    x2, y2 = nextNode
    if x2 > x1:
        return 0
    elif x2< x1:
        return 1
    elif y2>y1:
        return 2
    elif y2<y1:
        return 3
    


def find_maze_solution():
    cap = cv2.VideoCapture(0)

    # Capture initial frame to grab maze walls only once
    for i in range(5):
        ret, baseFrame = cap.read()
        i += 1

    
    frame2 = baseFrame.copy()
    xyIMG = pp.track_laser(frame2)
    cv2.imwrite('raw.png', baseFrame)
    redCoords, baseFrame = pp.track_red_dot(baseFrame, True, True)
    blueCoords, baseFrame = pp.track_blue_dot(baseFrame, True, True)
    pathFrame = pp.black_outside_borders(baseFrame)
    cv2.imwrite(r'output\\baseFrame.png', baseFrame)
    binMask = pp.get_binary(baseFrame)
    cv2.imwrite(r'output\\binMask.png', binMask)
    nodes, edges, nodesFrame = pp.create_nodes_for_pathfinding(pathFrame, 50, 50)
    cv2.imwrite(r'output\nodes.png', nodesFrame)
    startNode = pp.find_node_in_area(blueCoords, nodes)
    endNode = pp.find_node_in_area(redCoords, nodes)
    graph = create_graph(nodes, edges)
    path = dijkstra(graph, startNode, endNode)

    for i in range(1, len(path)):
        cv2.circle(pathFrame, path[i], radius=2, color=(0, 255, 0), thickness=-1)
        cv2.line(pathFrame, path[i - 1], path[i], color=(255, 0, 0), thickness=1)

    cv2.imwrite(r'output\pathFrame.png', pathFrame)


    cap.release()
    cv2.destroyAllWindows()

    return path, xyIMG[0]
