# related to DSP
import math
import json
import random
from matplotlib import pyplot as plt

class SplineInterval:
    def __init__(self, id, min_t, max_t, points, priority=0):
        self.min_t = min_t
        self.max_t = max_t
        self.points = sorted(points)
        self.updated_points = sorted(points)
        self.id = id
        self.priority = priority
    def add_point(self, point_t, updated=False):
        if updated:
            if point_t < self.points[0]:
                self.points.insert(0, point_t)
                return
            elif point_t > self.points[-1]:
                self.points.append(point_t)
                return
            for i in range(1, len(self.points)):
                if point_t < self.points[i]:
                    self.points.insert(i, point_t)
                    return
        else:
            if point_t < self.updated_points[0]:
                self.updated_points.insert(0, point_t)
                return
            elif point_t > self.updated_points[-1]:
                self.updated_points.append(point_t)
                return
            for i in range(1, len(self.updated_points)):
                if point_t < self.updated_points[i]:
                    self.updated_points.insert(i, point_t)
                    return
    def intersect(self, other):
        if self.min_t <= other.max_t and other.min_t <= self.max_t:
            return True
        else:
            return False

    def __contains__(self, t):
        if t >= self.min_t and t <= self.max_t:
            return True
        else:
            return False


class ControlPoints:
    def __init__(self, t, x, owner):
        self.t = t
        self.x = [x]
        self.owners = [owner]
        self.intersected = []

    def get_x(self, id):
        if len(self.owners) == 1:
            return self.x[0]
        else:
            for i in range(0, len(self.owners)):
                if self.owners[i] == id:
                    return self.x[i]

    def sum_x(self):
        out = 0
        for i in range(0, len(self.x)):
            out += self.x[i]
        return out

    def mean_x(self):
        out = 0
        for i in range(0, len(self.x)):
            out += self.x[i]
        return out / len(self.x)

    def update_x(self, new_x):
        self.x = new_x

    def update_x(self, new_t):
        self.t = new_t

    def add_owner(self, new_owner, new_x):
        self.owners.append(new_owner)
        self.x.append(new_x)

    def remove_owner(self, owner, x):
        self.owners.remove(owner)
        self.owners.remove(x)

    def add_intersected(self, new_intersected):
        self.intersected.append(new_intersected)

    def remove_intersected(self, intersected):
        self.intersected.remove(intersected)

    def __str__(self):
        out = "===========\nt = {}\n".format(self.t)
        out_x = "x and its owners are = ["
        for i in range(0, len(self.x)):
            out_x += str(self.x[i]) + " --> " + str(self.owners[i]) + ", "
        out_intersect = "]\nintersect with = ["
        for i in range(0, len(self.intersected)):
            out_intersect += str(self.intersected[i]) + " ,"
        return out + out_x + out_intersect + "]"


class ImpluseSpineCurveModel:
    def __init__(self, base=None):
        if base is None:
            self.control_points = {}
            self.impulse_intervals = []
            self.interval_count = 0
            self.control_pts_refined = []
            self.cached_xt = []
            self.cached_x = []
    def _interpolate(self, arr_t, t, interval_id, print_stff=False):
        if t <= arr_t[0]:
            return self.control_points[arr_t[0]].get_x(interval_id)
        if t >= arr_t[-1]:
            return self.control_points[arr_t[-1]].get_x(interval_id)
        for i in range(0, len(arr_t) - 1):
            if arr_t[i] <= t and arr_t[i + 1] > t:
                interval_index = i
                break
        x0 = self.control_points[arr_t[interval_index]].get_x(interval_id)
        x1 = self.control_points[arr_t[interval_index + 1]].get_x(interval_id)
        rtv = x0 + (t - arr_t[interval_index]) / (arr_t[interval_index + 1] - arr_t[interval_index]) * (x1 - x0)
        if print_stff:
            print(t, x0, x1, arr_t[interval_index], arr_t[interval_index + 1])

        return rtv
    def add_to_interval_tier(self, points, priority):
        # points are expected to be [[time], [space], priority] of shape [n,], [n,], [1,]
        # transition_interval is be of shape [2, ] such that the transition interval is within
        # time[x], time[y]
        t, x = points
        this_interval = []
        min_t = 100000
        max_t = -100000
        for i in range(0, len(t)):
            min_t = min(min_t, t[i])
            max_t = max(max_t, t[i])
            try:
                # add the interval as an owner to an existing point
                existant_ctpt = self.control_points[t[i]]
                existant_ctpt.add_owner(self.interval_count, x[i])
                this_interval.append(t[i])
            except:
                # add a new control point to the interval
                new_ctpt = ControlPoints(t[i], x[i], self.interval_count)
                this_interval.append(t[i])
                self.control_points[t[i]] = new_ctpt
        new_interval = SplineInterval(self.interval_count, min_t, max_t, this_interval, priority)
        self.impulse_intervals.append(new_interval)
        # update this index
        self.interval_count += 1
        return
    def fill_holes(self):
        # call this to create intervals that fill holes in the list of intervals, so that the base additive interval
        # always have a non-zero value, and other tiers can be added to this
        sorted_interval_list = sorted(self.impulse_intervals, key=lambda x: x.min_t)
        for i in range(0, len(sorted_interval_list) - 1):
            current_interval = sorted_interval_list[i]
            next_interval = sorted_interval_list[i + 1]
            if current_interval.max_t >= next_interval.min_t:
                pass
            else:
                start_v = self.control_points[current_interval.max_t].mean_x()
                end_v = self.control_points[next_interval.min_t].mean_x()
                self.add_to_interval_tier([[current_interval.max_t, next_interval.min_t], [start_v, end_v]], 0)
    def recompute_all_points_pre_process(self, mean=False):
        for i in range(self.interval_count):
            # iterate through the interval to see if the point exist in it
            if self.impulse_intervals[i].priority == 5:
                for j in range(self.interval_count):
                    if self.impulse_intervals[i].intersect(self.impulse_intervals[j]):
                        other_interval = self.impulse_intervals[j]
                        if other_interval.priority < 5 and other_interval.priority > 0:
                            for pt_t in other_interval.points:
                                for pt_i in range(len(self.control_points[pt_t].x)):
                                    if self.control_points[pt_t].owners[pt_i] == j:
                                        self.control_points[pt_t].x[pt_i] *= 0.1
    def recompute_all_points(self, mean=False):
        sorted_interval_list = sorted(self.impulse_intervals, key=lambda x: x.max_t)
        # for each point, check if they exist in any interval that does not own it. i.e. find interval it intersects
        for pt_t in self.control_points:
            added = False
            # iterate through the interval to see if the point exist in it
            for i in range(self.interval_count):
                interval = sorted_interval_list[i]
                if pt_t in interval:
                    added = True
                    if (not interval.id in self.control_points[pt_t].owners
                            and not interval.id in self.control_points[pt_t].intersected):
                        self.control_points[pt_t].add_intersected(interval.id)
                        self.impulse_intervals[interval.id].add_point(pt_t)
                elif pt_t > interval.max_t and added:
                    break
                else:
                    continue
        # if the intersection handling is to sum the signal
        self.control_pts_refined = []
        refined_t = []
        refined_x = []
        # sum up the control points
        for pt_t in self.control_points:
            pt = self.control_points[pt_t]
            additive_tier_value = 0
            additive_tier_count = 0
            mean_tier_value = 0
            mean_tier_count = 0
            override_tier_value = 0
            override_tier_count = 0
            override_tier_owner_count = 0
            # obtaining value of each owner and intersection:
            for owner in pt.owners:
                priority = self.impulse_intervals[owner].priority
                if priority == 0:
                    additive_tier_value += pt.get_x(owner)
                    additive_tier_count += 1
                elif priority == 5:
                    # if there are multiple priority 5 intervals, we calculate a mean
                    override_tier_value += pt.get_x(owner)
                    override_tier_count += 1
                    override_tier_owner_count += 1
                else:
                    mean_tier_value += pt.get_x(owner)
                    mean_tier_count += 1
            # if interval_ignored:
            #     continue
            for intersected_i in range(len(pt.intersected)):
                priority = self.impulse_intervals[pt.intersected[intersected_i]].priority
                if priority == 0:
                    additive_tier_value += self._interpolate(
                        self.impulse_intervals[pt.intersected[intersected_i]].points, pt_t,
                        pt.intersected[intersected_i])
                    additive_tier_count += 1
                elif priority == 5:
                    # if there are multiple priority 5 intervals, we calculate a mean
                    override_tier_value += self._interpolate(
                        self.impulse_intervals[pt.intersected[intersected_i]].points, pt_t,
                        pt.intersected[intersected_i])
                    override_tier_count += 1
                else:
                    mean_tier_value += self._interpolate(self.impulse_intervals[pt.intersected[intersected_i]].points,
                                                         pt_t, pt.intersected[intersected_i])
                    mean_tier_count += 1
            # compute mean for each type of value

            if additive_tier_count > 0:
                additive_tier_value = additive_tier_value / additive_tier_count
            if mean_tier_count > 0:
                mean_tier_value = mean_tier_value / mean_tier_count
            if override_tier_count > 0:
                override_tier_value = override_tier_value / override_tier_count

            # if there is an override tier, ignore all other tiers
            if override_tier_count > 0 and override_tier_owner_count == 0:
                pass
            elif override_tier_count > 0:
                refined_x.append(override_tier_value)
                refined_t.append(pt_t)
            else:
                refined_x.append(mean_tier_value + additive_tier_value)
                refined_t.append(pt_t)
        # sort the two lists
        refined_x = [x for _, x in sorted(zip(refined_t, refined_x), key=lambda pair: pair[0])]
        refined_t = sorted(refined_t)

        # cache the result and output
        self.cached_x = refined_x
        self.cached_xt = refined_t
        return [refined_t, refined_x]
    def sum_with_other_tiers(self, others, eat=False):
        # this sums the current tier with other tiers that shows up
        new_tier = ImpluseSpineCurveModel()
        others.append(self)
        for tier in others:
            for interval in tier.impulse_intervals:
                point_set_x = []
                point_set_t = []
                for pt in interval.points:
                    point_set_x.append(tier.control_points[pt].get_x(interval.id))
                    point_set_t.append(pt)
                new_tier.add_to_interval_tier([point_set_t, point_set_x], interval.priority)
        if eat:
            new_tier.recompute_all_points_pre_process()
        return new_tier.recompute_all_points()
if __name__ == "__main__":
    tiers = ImpluseSpineCurveModel()
    sentence_level_ts = [[-0.03, 0.159, 0.31999999999999995], [0.31999999999999995, 1.73], [3.11, 3.3529999999999998, 3.56], [3.56, 4.119999999999999], [4.76, 4.949, 5.109999999999999], [5.109999999999999, 6.3999999999999995], [6.8, 7.0], [8.17, 8.413, 8.62], [8.62], [9.25, 9.493, 9.7], [9.7, 10.639999999999999], [11.92, 12.163, 12.37], [12.37, 12.77], [13.700000000000001, 13.889000000000001, 14.05], [14.05, 14.73], [14.930000000000001, 15.173000000000002, 15.38], [15.38, 15.950000000000001, 18.35]]
    sentence_level_xs = [[0.0, -0.47189530806467206, -0.58986913508084], [-0.58986913508084, -0.47189530806467206], [-0.412908394556588, 5.874894289300162, 7.4468449602643485], [7.4468449602643485, 5.9574759682114795], [5.212791472185043, 8.777674274736944, 9.668894975374918], [9.668894975374918, 7.735115980299935], [9.668894975374918, 7.735115980299935], [6.768226482762443, -0.7461778683963329, -2.6247789561860264], [-2.6247789561860264], [-1.8373452693302184, 6.1647680058495125, 8.165296324644446], [8.165296324644446, 6.532237059715557], [5.715707427251112, 2.2335430423923297, 1.3630019461776346], [1.3630019461776346, 1.0904015569421077], [0.9541013623243442, 5.159444580706199, 6.210780385301662], [6.210780385301662, 4.96862430824133], [4.347546269711163, -3.4936558915201594, -5.45395643182799], [-5.45395643182799, -4.363165145462392, -4.363165145462392]]
    sentence_level_priority = [5, 0, 5, 0, 5, 0, 5, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0]
    impulse_xs = [[0, 1.5, -1.5, -1.2000000000000002, 0], [0, 1.5, -1.5, -1.2000000000000002, 0], [0, 1.2, -2.8, -2.2399999999999998, 0], [0, 1.2, -2.8, -2.2399999999999998, 0], [0, 1.5, -1.5, -1.2000000000000002, 0], [0, 1.5, -1.5, -1.2000000000000002, 0], [0, 1.2, -2.8, -2.2399999999999998, 0], [0, 1.2, -2.8, -2.2399999999999998, 0], [0, 1.2, -2.8, -2.2399999999999998, 0]]
    impulse_ts = [[0.49000000000000005, 0.72, 0.97, 1.295, 1.82], [3.79, 3.87, 4.12, 4.32, 4.72], [5.76, 6.03, 6.28, 6.54, 7.0], [5.76, 6.03, 6.28, 6.54, 7.0], [6.7, 7.685, 8.37, 8.42, 8.67], [10.62, 10.7, 10.95, 10.995, 11.239999999999998], [11.819999999999999, 11.96, 12.21, 12.305, 12.6], [12.87, 13.535, 13.9, 13.975000000000001, 14.25], [14.83, 15.2, 15.45, 15.565, 15.879999999999999]]
    impulse_priority = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    jitter_level_ts = [[3.05, 3.1999999999999997, 3.4, 3.7], [4.61, 4.74, 4.96, 5.26], [5.630000000000001, 5.78, 5.98, 6.26], [9.13, 9.280000000000001, 9.48, 9.809999999999999], [10.030000000000001, 10.180000000000001, 10.38, 10.629999999999999], [13.81, 13.96, 14.16, 14.43], [15.58, 15.73, 15.93, 16.21]]
    jitter_level_xs = [[0, 0.3, -0.3, 0], [0, 0.3, -0.3, 0], [0, 0.3, -0.3, 0], [0, 0.3, -0.3, 0], [0, 0.3, -0.3, 0], [0, 0.3, -0.3, 0], [0, 0.3, -0.3, 0]]
    jitter_level_priority = [1, 1, 1, 1, 1, 1, 1]
    # ts = [jitter_level_ts, impulse_ts, sentence_level_ts]
    # xs = [jitter_level_xs, impulse_xs, sentence_level_xs]
    sentence_tiers = ImpluseSpineCurveModel()
    for i in range(0, len(jitter_level_ts)):
        tiers.add_to_interval_tier([jitter_level_ts[i], jitter_level_xs[i]], jitter_level_priority[i])
    for i in range(0, len(impulse_xs)):
        tiers.add_to_interval_tier([impulse_ts[i], impulse_xs[i]], impulse_priority[i])
    for i in range(0, len(sentence_level_ts)):
        sentence_tiers.add_to_interval_tier([sentence_level_ts[i], sentence_level_xs[i]], sentence_level_priority[i])

    sentence_tiers.fill_holes()
    t, x = tiers.sum_with_other_tiers([sentence_tiers])
    # t2, x2 = tiers.sum_with_other_tier([tiers])
    # x = [x for _, x in sorted(zip(t, x))]
    # t = sorted(t)
    plt.subplot(3,1,2)
    plt.plot(t, x)
    # for i in range(len(impulse_ts)):
    #     plt.plot(impulse_ts[i], impulse_xs[i])
    plt.subplot(3, 1, 1)
    out_t = []
    out_x = []
    out_priority = []
    for item in sentence_tiers.impulse_intervals:
        pts = item.points
        temp_t = []
        temp_x = []
        temp_priority = []
        for pt in pts:
            temp_t.append(pt)
            point = sentence_tiers.control_points[pt]
            temp_x.append(point.get_x(item.id))
        out_priority.append(item.priority)
        out_t.append(temp_t)
        out_x.append(temp_x)
    for i in range(len(out_x)):
        plt.plot(out_t[i], out_x[i])
    t, x = tiers.sum_with_other_tiers([sentence_tiers], eat=True)
    plt.subplot(3, 1, 3)
    plt.plot(t, x)
    plt.show()