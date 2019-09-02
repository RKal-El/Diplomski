''' PID (Proportional-Integral-Derivative) Controller

Used to keep speed close to a set speed.

Marko Rasetina
July 31, 2019
'''


class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.speed = 0.
        self.errors = [0] * 29

    def set_speed(self, speed):
        self.speed = speed

    def update(self, measurement):
        self.errors.append(self.speed - measurement)

        output = (self.Kp * self.errors[-1] +
                  self.Ki * sum(self.errors) +
                  self.Kd * self.errors[-2])

        if (output > -0.3) and (output < 0.0):
            output = 0.0

        self.errors.pop(0)

        return output
