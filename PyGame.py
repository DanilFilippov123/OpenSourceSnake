import csv

import numpy
import numpy as np
import pygame
import random
from pygame.math import Vector2

import cv2
import mediapipe as mp
import os


pygame.init()
 
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)

dis = pygame.display.set_mode(flags=pygame.FULLSCREEN)

#dis_height, dis_width  = dis.get_size()

dis_width = 1920
dis_height = 1080
#
# dis = pygame.display.set_mode((dis_width, dis_height))


pygame.display.set_caption('Snake Game by Pythonist')
 
clock = pygame.time.Clock()
 
snake_block = 40
snake_speed = 15
framerate = 60

side_bar_width = round(dis_width / 3)
side_bar_height = 480

game_zone_height = 400
game_zone_width = 400

left_padding = (side_bar_width - game_zone_width) // 2
bottom_padding = (side_bar_height - game_zone_height) // 2

game_screen_rect = pygame.Rect(side_bar_width, 0, dis_width - side_bar_width, dis_height)

prev_cell_coof = 50

font_style = pygame.font.SysFont("bahnschrift", 45)
score_font = pygame.font.SysFont("comicsansms", 35)

score_file = "score.csv"
score_list: list[dict[str, int]] = []

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
        h, w, _ = image.shape
        image = Hands.resize(image)

        image = cv2.flip(image, flipCode=1)
        result = Hands.hands.process(image)

        if result.multi_hand_landmarks:
            Hands.draw.draw_landmarks(image, result.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
            for idx, finger in enumerate(result.multi_hand_landmarks[0].landmark):
                if idx == 8:
                    cx, cy = int(finger.x * w), int(finger.y * h)
                    return Vector2(cx, cy), image
        return None, image



    @staticmethod
    def close():
        Hands.hands.close()
        Hands.cap.release()


def your_score(score: int, surf: pygame.Surface):
    value = score_font.render("Нынешний счёт: " + str(score), True, yellow)
    surf.blit(value, (0, game_screen_rect.h - 400))
    score_list.sort(key=lambda x: x["score"], reverse=True)

    score_list_int = list(map(lambda x: x["score"], score_list))
    score_list_int = list(set(score_list_int))
    score_list_int.sort(reverse=True)

    txt = score_font.render("Топ:", True, yellow)
    surf.blit(txt, (60, game_screen_rect.h - 350))
    for i, s in enumerate(score_list_int[:5]):
        value = score_font.render(" " + str(s), True, yellow)
        surf.blit(value, (60, game_screen_rect.h - 320 + i * 40 + 20))

 
 
def our_snake(snake_head: pygame.Rect, snake_list: list[pygame.Rect]):
    for x in snake_list:
        pygame.draw.circle(dis, red, x.center, snake_block / 2)
    pygame.draw.rect(dis, white, snake_head)

 
def message(msg: str, pos: tuple[int, int], color):
    mess = font_style.render(msg, True, color)
    dis.blit(mess, [pos[0], pos[1]])


def quite_game(score):
    Hands.close()
    pygame.quit()
    score_list.append({"score": score})
    save_score(score_list, score_file)
    quit()

def side_bar_surface(image, score) -> pygame.Surface:
    surface = pygame.Surface((side_bar_width, dis_height))
    surface.fill((0,0,0))
    surface.blit(Hands.cv2_image_to_surface(image), (0, 0))
    your_score(score, surface)
    return surface

def load_score(file_name: str) -> list[dict[str, int]]:
    with open(file_name) as f:
        reader = csv.DictReader(f)
        res = []
        row: dict[str, int]
        for row in reader:
            row["score"] = int(row["score"])
            res.append(row)
        return res
def save_score(score_list: list[dict[str, int]], file_name: str):
    with open(file_name, "w") as f:
        writer = csv.DictWriter(f, ("score",))
        writer.writeheader()
        score_list.sort(key=lambda x: x["score"], reverse=True)
        writer.writerows(score_list)


def game_loop():
    head_vec = Vector2(round(random.randrange(side_bar_width, dis_width - snake_block) / 10.0) * 10.0,
                       round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0)
    head_rect = pygame.Rect(head_vec, (snake_block, snake_block))

    snake_list: list[pygame.Rect] = [head_rect]

    food_vec = Vector2(round(random.randrange(side_bar_width, dis_width - snake_block) / 10.0) * 10.0,
                       round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0)
    food_rect = pygame.Rect(food_vec, (snake_block, snake_block))

    snake_speed_vec = Vector2(0, 0)

    game_close = False

    score = 0

    update_snake_event = pygame.USEREVENT + 1
    pygame.time.set_timer(update_snake_event, 93)

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
            message("Вы проиграли! Нажмите Q чтобы выйти или C чтобы продолжить.", (dis_width // 3, dis_height // 2), red)
            _, image = Hands.cv2_finger_coord_image()
            dis.blit(side_bar_surface(image, score), (0, 0))
            pygame.display.update()

            while game_close:
                _, image = Hands.cv2_finger_coord_image()
                dis.blit(side_bar_surface(image, score), (0, 0))
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            quite_game(score)
                        if event.key == pygame.K_c:
                            re_init_game()
                            game_close = False
                            break
                        if event.type == pygame.QUIT:
                            quite_game(score)
                pygame.display.update()



        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quite_game(score)
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
                head_vec = Vector2(head_rect.center)
                snake_list.append(head_rect.copy())
                if len(snake_list) > score + 1:
                    del snake_list[0]


                if food_rect.colliderect(head_rect):
                    score += 1
                    food_vec = Vector2(round(random.randrange(side_bar_width, dis_width - snake_block) / 10.0) * 10.0,
                                       round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0)
                    food_rect = pygame.Rect(food_vec, (snake_block, snake_block))
                    while any((Vector2(cell.center).distance_to(food_vec) < snake_block for cell in snake_list)):
                        food_vec = Vector2(
                            round(random.randrange(side_bar_width, dis_width - snake_block) // snake_block) * snake_block,
                            round(random.randrange(0, dis_height - snake_block) // snake_block) * snake_block)
                        food_rect = pygame.Rect(food_vec, (snake_block, snake_block))
                if any((Vector2(cell.center).distance_to(head_vec) < 10 for cell in snake_list[:-1])):
                    score_list.append({"score": score})
                    game_close = True

        finger_vec_new, image = Hands.cv2_finger_coord_image()



        dis.fill(blue)

        if finger_vec_new is not None:
            finger_vec_new.x = ((finger_vec_new.x - left_padding) / game_zone_width) * dis_width + side_bar_width
            if finger_vec_new.x < side_bar_width:
                finger_vec_new.x = side_bar_width + 10
            if finger_vec_new.x > dis_width:
                finger_vec_new.x = dis_width

            finger_vec_new.y = ((finger_vec_new.y - bottom_padding) / game_zone_height) * dis_height
            if finger_vec_new.y < 0:
                finger_vec_new.y = 10
            if finger_vec_new.y > dis_height:
                finger_vec_new.y = dis_height - 10

            pygame.draw.circle(dis, green, finger_vec_new, 10)


            snake_speed_vec =  (finger_vec_new - head_vec).normalize() * snake_speed


        image = cv2.rectangle(image, (int(left_padding), int(bottom_padding)), (int(side_bar_width - left_padding), int(side_bar_height - bottom_padding)), green)


        dis.blit(side_bar_surface(image, score), (0,0))




        if not game_screen_rect.contains(head_rect):
            score_list.append({"score": score})
            game_close = True

        pygame.draw.rect(dis, green, food_rect)

        our_snake(head_rect, snake_list)

        pygame.draw.line(dis, black, head_vec, head_vec + snake_speed_vec)
        pygame.draw.circle(dis, black, head_vec + snake_speed_vec, 2)

        pygame.display.update()

        clock.tick(framerate)

if __name__ == "__main__":
    if os.path.isfile("score.csv"):
        score_list = load_score("score.csv")
    else:
        score_list = []
    game_loop()