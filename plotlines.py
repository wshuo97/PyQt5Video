# from pathlib import WindowsPath
import argparse
import cv2


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    tl = line_thickness or round(
        0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        # print("label : ",label)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def loadres(filepath):
    print(filepath)
    results = []
    maxframe = 0
    maxlabel = 0
    with open(filepath, "r") as f:
        for line in f:
            lines = line.strip().split(",")[0:-4]
            lines = [float(v) for v in lines]
            frame, label, x, y, w, h = (int(x) for x in lines)
            maxframe = max(maxframe, frame+10)
            maxlabel = max(maxlabel, label+10)
            results.append((frame, label, x, y, w, h))
    frameret = [None]*maxframe
    labelret = [None]*maxlabel
    for (frame, label, x, y, w, h) in results:
        if frameret[frame] is None:
            frameret[frame] = list()
        if labelret[label] is None:
            labelret[label] = list()
        frameret[frame].append(label)
        labelret[label].append((x, y, w, h, frame))
    return (frameret, labelret, maxlabel)


def drawline(fret, lret, maxl, source, outpath, predtime):
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(
        "".join(outpath), cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), fps, size)
    frameid = 0
    lcnt = [0 for _ in range(maxl)]
    print(cap.isOpened())
    colors = Colors()
    while cap.isOpened():
        _, frame = cap.read()
        if frame is None:
            break
        frameid = frameid+1
        # if frameid < 5190 or frameid > 5260:
        #     continue
        if fret[frameid] is None:
            continue
        for idx in fret[frameid]:
            # if idx != 1213:
            #     continue
            lcnt[idx] = lcnt[idx]+1
            stindex = lcnt[idx]-10
            if stindex < 0:
                stindex = 0
            # print()
            px, py, pw, ph, f = lret[idx][stindex]
            px, py = int(px+pw/2), int(py+ph/2)
            # cnt = stindex
            color = colors(idx % 255, True)

            th = 0
            maxv = 0
            for (x, y, w, h, f) in lret[idx][stindex:lcnt[idx]]:
                cx, cy = int(x+w/2), int(y+h/2)
                th = th+abs(px-cx)+abs(py-cy)
                maxv = max(abs(px-cx)+abs(py-cy), maxv)
                px, py = cx, cy
            th = th/(lcnt[idx]-stindex)
            # print(th, maxv)
            if th*2 < maxv:
                th = th*2
            else:
                th = maxv+1
            # import pdb;pdb.set_trace()
            px, py, pw, ph, pf = lret[idx][stindex]
            px, py = int(px+pw/2), int(py+ph/2)

            vx, vy = 0.0, 0.0
            nx, ny, w, h = px, py, pw, ph
            # print(lcnt[idx])
            lens = abs(pf-lret[idx][lcnt[idx]-1][-1])
            if stindex+1 == lcnt[idx]:
                continue
            for (x, y, w, h, f) in lret[idx][stindex+1:lcnt[idx]]:
                cx, cy = int(x+w/2), int(y+h/2)
                if abs(px-cx)+abs(py-cy) < th:
                    vx = vx+(cx-px)
                    vy = vy+(cy-py)
                    cv2.line(frame, (px, py), (cx, cy), color, 3)
                    px, py = cx, cy
                    nx, ny, w, h = px, py, w, h
                    
            plot_one_box((int(nx-w/2), int(ny-h/2),
                         int(nx+w/2), int(ny+h/2)), frame, color, label=str(idx))
            if lens == 0:
                lens = 10
            # print(lens)
            vx, vy = int(vx/lens), int(vy/lens)

            for _ in range(int(predtime*fps)):
                nx = nx+vx
                ny = ny+vy
                cx, cy = int(nx), int(ny)
                cv2.line(frame, (px, py), (cx, cy), color, 3)
                px, py = nx, ny

        print(
            "frame : ({}), tracklets : {}".format(frameid, len(fret[frameid])))
        videoWriter.write(frame)

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()
    # pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str, default='')
    parser.add_argument('--tracks', type=str, default='')
    parser.add_argument('--outpath', type=str, default='')
    parser.add_argument('--predtime', type=int, default=1)

    opt = parser.parse_args()

    a, b, c = loadres(opt.tracks)
    drawline(a, b, c, opt.source, opt.outpath, opt.predtime)
