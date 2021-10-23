from deep_sort.sort import tracker
import numpy as np
import cv2


class rectangle(object):
    def __init__(self, x=0., y=0., width=0., height=0., copyobj=None):
        if copyobj:
            self.x = copyobj.x
            self.y = copyobj.y
            self.width = copyobj.width
            self.height = copyobj.height
        else:
            self.x = x
            self.y = y
            self.width = width
            self.height = height


class particle(object):
    def __init__(self, orix=0, oriy=0, x=0, y=0, scale=0., prex=0., prey=0, prescale=0., rect=None, hist=None, weight=0, copyobj=None):
        if copyobj:
            self.orix = copyobj.orix
            self.oriy = copyobj.oriy
            self.x = copyobj.x
            self.y = copyobj.y
            self.scale = copyobj.scale
            self.prex = copyobj.prex
            self.prey = copyobj.prey
            self.prescale = copyobj.prescale
            self.rect = copyobj.rect
            self.hist = copyobj.hist
            self.weight = copyobj.weight
        else:
            self.orix = orix
            self.oriy = oriy
            self.x = x
            self.y = y
            self.scale = scale
            self.prex = prex
            self.prey = prey
            self.prescale = prescale
            self.rect = rect
            self.hist = hist
            self.weight = weight

    def printval(self, id):
        print("*", id, self.rect.x, self.rect.y,
              self.rect.width, self.rect.height, self.weight)


class ParticleFilter(object):
    def __init__(self, hist_size=[16, 16, 16], ranges=[0, 180, 0, 256, 0, 256], channels=[0, 1, 2],
                 A1=2, A2=-1, B0=1, sigmax=1.0, sigmay=0.5, sigmas=0.001, particle_number=100, frame_shape=(540, 960, 3)):
        self.hist_size = hist_size

        self.ranges = ranges

        self.channels = channels

        # about particle window
        self.A1 = A1
        self.A2 = A2
        self.B0 = B0
        self.sigmax = sigmax
        self.sigmay = sigmay
        self.sigmas = sigmas
        self.particle_number = particle_number
        self.frame_shape = frame_shape

        self.ori_rect = None
        self.ori_hist = None

        self.particles = []

    def create(self, bbox_tlwh, im):
        # import pdb;pdb.set_trace()
        bbox_tlwh = [int(x) for x in bbox_tlwh]
        select = rectangle(
            x=bbox_tlwh[0], y=bbox_tlwh[1], width=bbox_tlwh[2], height=bbox_tlwh[3])
        self.ori_rect = select
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        for i in range(self.particle_number):
            self.particles.append(particle())
        target_img = hsv[select.y:select.y +
                         select.height, select.x:select.x+select.width]
        target_hist = cv2.calcHist(
            [target_img], self.channels, None, self.hist_size, self.ranges)
        cv2.normalize(target_hist, target_hist)
        self.ori_hist = target_hist
        for i in range(self.particle_number):
            self.particles[i].x = int(select.x)
            self.particles[i].y = int(select.y)
            self.particles[i].orix = self.particles[i].x
            self.particles[i].oriy = self.particles[i].y
            self.particles[i].prex = self.particles[i].x
            self.particles[i].prey = self.particles[i].y
            self.particles[i].rect = rectangle(copyobj=select)
            self.particles[i].prescale = 1
            self.particles[i].scale = 1
            self.particles[i].hist = self.ori_hist
            self.particles[i].weight = 0

        self.age = 0
        # print("create, ", im.shape, select.x,
        #       select.y, select.width, select.height)

    def predict(self):
        recttrack = rectangle()
        mparticle = self.particles[0]
        recttrack.x = round(mparticle.rect.x)
        recttrack.y = round(mparticle.rect.y)
        recttrack.width = mparticle.rect.width
        recttrack.height = mparticle.rect.height

        # return coordinate format : x,y,x,y
        return recttrack.x, recttrack.y, recttrack.x+recttrack.width, recttrack.y+recttrack.height
        # pass

    def update(self, cur_rect=None, im=None):
        self.age += 1
        cur_rect = [int(x) for x in cur_rect]
        track_rect = rectangle(x=cur_rect[0], y=cur_rect[1],
                               width=cur_rect[2], height=cur_rect[3])
        sums = 0.0
        rng = np.random
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        target_hist = self.ori_hist

        # print("track_rect : {} {} {} {}".format(track_rect.x,
        #       track_rect.y, track_rect.width, track_rect.height))

        # target_hist = [0.0]

        if track_rect is not None:
            track_img1 = hsv[track_rect.y:track_rect.y +
                             track_rect.height, track_rect.x:track_rect.x+track_rect.width]
            target_hist = cv2.calcHist(
                [track_img1], self.channels, None, self.hist_size, self.ranges)
            cv2.normalize(target_hist, target_hist)
            self.ori_hist = target_hist
        # else:
        #     target_hist = self.ori_hist

        for i in range(self.particle_number):
            xpre = self.particles[i].x
            ypre = self.particles[i].y
            pres = self.particles[i].scale

            # update the rect
            x = round(self.A1*(self.particles[i].x-self.particles[i].orix)+self.A2*(self.particles[i].prex -
                                                                                    self.particles[i].orix)+self.B0*rng.normal(scale=self.sigmax)+self.particles[i].orix)
            self.particles[i].x = max(0, min(x, self.frame_shape[1]-1))

            y = round(self.A1*(self.particles[i].y-self.particles[i].oriy)+self.A2*(self.particles[i].prey -
                                                                                    self.particles[i].oriy)+self.B0*rng.normal(scale=self.sigmay)+self.particles[i].oriy)
            self.particles[i].y = max(0, min(y, self.frame_shape[0]-1))

            s = self.A1*(self.particles[i].scale-1)+self.A2 * \
                (self.particles[i].prescale-1) + \
                self.B0*(rng.normal(scale=self.sigmas))+1.0
            self.particles[i].scale = max(1.0, min(s, 3.0))

            self.particles[i].prex = xpre
            self.particles[i].prey = ypre
            self.particles[i].prescale = pres
            rect = self.particles[i].rect

            self.particles[i].rect = rectangle()

            self.particles[i].rect.x = max(0, min(round(
                self.particles[i].x-0.1424*self.particles[i].scale*rect.width), self.frame_shape[1]))
            self.particles[i].rect.y = max(0, min(round(
                self.particles[i].y-0.1424*self.particles[i].scale*rect.height), self.frame_shape[0]))
            self.particles[i].rect.width = min(
                round(rect.width), self.frame_shape[1]-rect.x)
            self.particles[i].rect.height = min(
                round(rect.height), self.frame_shape[0]-rect.y)

            rect1 = self.particles[i].rect
            track_img_ = hsv[rect1.y:rect1.y +
                             rect1.height, rect1.x:rect1.x+rect1.width]
            track_hist = cv2.calcHist(
                [track_img_], self.channels, None, self.hist_size, self.ranges)
            cv2.normalize(track_hist, track_hist)
            self.particles[i].weight = 1.0 - \
                cv2.compareHist(target_hist, track_hist,
                                cv2.HISTCMP_BHATTACHARYYA)
            sums = sums+self.particles[i].weight
            # print(self.particles[i].weight)
        if sums < 1e-6:
            self.refresh(track_rect)
            # import pdb
            # pdb.set_trace()
        else:
            for i in range(self.particle_number):
                # self.particles[i].printval(i)
                self.particles[i].weight = self.particles[i].weight/sums

    def refresh(self, select):
        for i in range(self.particle_number):
            self.particles[i].x = int(select.x)
            self.particles[i].y = int(select.y)
            self.particles[i].orix = self.particles[i].x
            self.particles[i].oriy = self.particles[i].y
            self.particles[i].prex = self.particles[i].x
            self.particles[i].prey = self.particles[i].y
            self.particles[i].rect = rectangle(copyobj=select)
            self.particles[i].prescale = 1
            self.particles[i].scale = 1
            self.particles[i].hist = self.ori_hist
            self.particles[i].weight = 0

        self.age = 0

    def resample(self):
        sparticles = sorted(
            self.particles, key=lambda x: x.weight, reverse=True)
        k = 0
        nparticles = list()
        for i in range(self.particle_number):
            npv = round(sparticles[i].weight*self.particle_number)
            for j in range(round(npv)):
                nparticles.append(particle(copyobj=sparticles[i]))
                k = k+1
                if k >= self.particle_number:
                    break
            if k >= self.particle_number:
                break
        while k < self.particle_number:
            k = k+1
            nparticles.append(particle(copyobj=sparticles[0]))
        for i in range(self.particle_number):
            self.particles[i] = nparticles[i]
