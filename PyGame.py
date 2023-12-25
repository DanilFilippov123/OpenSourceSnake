import numpy
import numpy as np
import pygame
import random
from pygame.math import Vector2

import cv2
import mediapipe as mp

pygame.init()
 
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)
 
dis_width = 1080
dis_height = 960
 
dis = pygame.display.set_mode((dis_width, dis_height))
pygame.display.set_caption('Snake Game by Pythonist')
 
clock = pygame.time.Clock()
 
snake_block = 80
snake_speed = snake_block
framerate = 60

side_bar_width = round(dis_width / 3)
side_bar_height = dis_height / 3

game_screen_rect = pygame.Rect(side_bar_width, 0, dis_width - side_bar_width, dis_height)

prev_cell_coof = 50

font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)
 
class Hands:
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    @staticmethod
    def cv2_image_to_surface(cv_image: np.ndarray) -> pygame.Surface:
        if cv_image.dtype.name == 'uint16':
            cv_image = (cv_image / 256).astype('uint8')
        size = cv_image.shape[1::-1]
        if len(cv_image.shape) == 2:
            cv_image = np.repeat(cv_image.reshape(size[1], size[0], 1), 3, axis=2)
            format_pixel = 'RGB'
        else:
            format_pixel = 'RGBA' if cv_image.shape[2] == 4 else 'RGB'
            cv_image[:, :, [0, 2]] = cv_image[:, :, [2, 0]]
        surface = pygame.image.frombuffer(cv_image.flatten(), size, format_pixel)
        return surface.convert_alpha() if format_pixel == 'RGBA' else surface.convert()

    @staticmethod
    def resize(image):
        if image.shape[1] > side_bar_width:
             coefficient = side_bar_width / image.shape[1]
             width = int(image.shape[1] * coefficient)
             height = int(image.shape[0] * coefficient)
             size = (width, height)
             return cv2.resize(image, size)
        return image

    @staticmethod
    def cv2_finger_coord_image() -> tuple[Vector2 | None, numpy.ndarray]:
        _, image = Hands.cap.read()
        image = Hands.resize(image)
        h, w, _ = image.shape
        result = Hands.hands.process(image)
        if result.multi_hand_landmarks:
            Hands.draw.draw_landmarks(image, result.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
            for idx, finger in enumerate(result.multi_hand_landmarks[0].landmark):
                return Vector2(finger.x * h, finger.y * w), image
        return None, image



    @staticmethod
    def close():
        Hands.hands.close()
        Hands.cap.release()


def your_score(score):
    value = score_font.render("Your Score: " + str(score), True, yellow)
    dis.blit(value, [0, game_screen_rect.h - 200])
 
 
 
def our_snake(snake_head: pygame.Rect, snake_list: list[pygame.Rect]):
    for x in snake_list:
        pygame.draw.circle(dis, red, x.center, snake_block / 2)
    pygame.draw.rect(dis, white, snake_head)

 
def message(msg, color):
    mess = font_style.render(msg, True, color)
    dis.blit(mess, [dis_width / 6, dis_height / 3])


def quite_game():
    Hands.close()
    pygame.quit()
    quit()

def game_loop():
    head_vec = Vector2((dis_width + side_bar_width) / 2, dis_width / 2)
    head_rect = pygame.Rect(head_vec, (snake_block, snake_block))

    snake_list: list[pygame.Rect] = [head_rect]

    food_vec = Vector2(round(random.randrange(side_bar_width, dis_width - snake_block) / 10.0) * 10.0,
                       round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0)
    food_rect = pygame.Rect(food_vec, (snake_block, snake_block))

    snake_speed_vec = Vector2(0, 0)

    game_close = False

    score = 0

    update_snake_event = pygame.USEREVENT + 1
    pygame.time.set_timer(update_snake_event, 500)

    update_finger_coordinates = pygame.USEREVENT + 2
    pygame.time.set_timer(update_finger_coordinates, 1000)

    def re_init_game():
        nonlocal head_vec, snake_list, food_vec, head_rect, food_rect,snake_speed_vec,game_close,score
        head_vec = Vector2((dis_width + side_bar_width) / 2, dis_width / 2)
        head_rect = pygame.Rect(head_vec, (snake_block, snake_block))
        snake_list = [head_rect]
        food_vec = Vector2(round(random.randrange(side_bar_width, dis_width - snake_block) / 10.0) * 10.0,
                           round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0)
        food_rect = pygame.Rect(food_vec, (snake_block, snake_block))
        snake_speed_vec = Vector2(0, 0)
        game_close = False
        score = 0

    while True:
        if game_close:
            dis.fill(blue)
            message("You Lost! Press C-Play Again or Q-Quit", red)
            your_score(score)
            pygame.display.update()

            while game_close:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            quite_game()
                        if event.key == pygame.K_c:
                            re_init_game()
                            game_close = False
                            break

        finger_vec_new, image = Hands.cv2_finger_coord_image()
        if finger_vec_new is not None:
            finger_vec = finger_vec_new

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quite_game()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    snake_speed_vec.xy = -snake_speed, 0
                elif event.key == pygame.K_RIGHT:
                    snake_speed_vec.xy = snake_speed, 0
                elif event.key == pygame.K_UP:
                    snake_speed_vec.xy = 0, -snake_speed
                elif event.key == pygame.K_DOWN:
                    snake_speed_vec.xy = 0, snake_speed
            if event.type == update_snake_event:
                snake_speed_vec = snake_speed_vec.normalize() * snake_speed if snake_speed_vec.length() != 0 else snake_speed_vec

                head_rect.move_ip(snake_speed_vec)

                snake_list.append(head_rect.copy())
                if len(snake_list) > score + 1:
                    del snake_list[0]

                if food_rect.colliderect(head_rect):
                    score += 1
                    food_vec = Vector2(round(random.randrange(side_bar_width, dis_width - snake_block) / 10.0) * 10.0,
                                       round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0)
                    food_rect = pygame.Rect(food_vec, (snake_block, snake_block))

                if any((cell.contains(head_rect) for cell in snake_list[:-1])):
                    game_close = True

        dis.fill(blue)
        dis.blit(Hands.cv2_image_to_surface(image), (0, 0))

        if not game_screen_rect.contains(head_rect):
            game_close = True

        pygame.draw.rect(dis, green, food_rect)

        our_snake(head_rect, snake_list)
        your_score(score)

        pygame.display.update()

        clock.tick(framerate)

game_loop()