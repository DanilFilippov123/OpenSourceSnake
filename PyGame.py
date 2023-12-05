import numpy as np
import pygame
import time
import random

import cv2
import mediapipe as mp

pygame.init()
 
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)
 
dis_width = 640
dis_height = 480
 
dis = pygame.display.set_mode((dis_width, dis_height))
pygame.display.set_caption('Snake Game by Pythonist')
 
clock = pygame.time.Clock()
 
snake_block = 80
snake_speed = (80 / 60) * 1.6
framerate = 60

prev_cell_coof = 50

font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)
 
 
def your_score(score):
    value = score_font.render("Your Score: " + str(score), True, yellow)
    dis.blit(value, [0, 0])
 
 
 
def our_snake(snake_head, snake_list):
    pygame.draw.rect(dis, black, [snake_head[0], snake_head[1], snake_block, snake_block])
    for x in snake_list:
        pygame.draw.circle(dis, black, [x[0] + snake_block / 2, x[1] + snake_block / 2], snake_block / 2)
 
 
def message(msg, color):
    mesg = font_style.render(msg, True, color)
    dis.blit(mesg, [dis_width / 6, dis_height / 3])


def resize(image):
    if image.shape[1] > 900:
        coefficient = 900 / image.shape[1]
        width = int(image.shape[1] * coefficient)
        height = int(image.shape[0] * coefficient)
        size = (width, height)
        return cv2.resize(image, size)
    return image


def cv2_image_to_surface(cv_image: np.ndarray) -> pygame.Surface:
    if cv_image.dtype.name == 'uint16':
        cv_image = (cv_image / 256).astype('uint8')
    size = cv_image.shape[1::-1]
    if len(cv_image.shape) == 2:
        cv_image = np.repeat(cv_image.reshape(size[1], size[0], 1), 3, axis=2)
        format = 'RGB'
    else:
        format = 'RGBA' if cv_image.shape[2] == 4 else 'RGB'
        cv_image[:, :, [0, 2]] = cv_image[:, :, [2, 0]]
    surface = pygame.image.frombuffer(cv_image.flatten(), size, format)
    return surface.convert_alpha() if format == 'RGBA' else surface.convert()

def cv2_finger_coord_image(hands, cap) -> tuple[float, float, np.ndarray]:
    x1_change = 0
    y1_change = 0
    offset = 100

    image: np.ndarray | None = None
    if cap.isOpened():
        success, image = cap.read()
        image = resize(image)
        if not success:
            print("Ignoring empty camera frame.")
            return 0, 0, image
    h, w, _ = image.shape
    x_center = w // 2
    y_center = h // 2
    cv2.line(image, (x_center + offset, 0), (x_center + offset, h), green, 2)
    cv2.line(image, (x_center - offset, 0), (x_center - offset, h), green, 2)
    cv2.line(image, (0, y_center + offset), (w, y_center + offset), green, 2)
    cv2.line(image, (0, y_center - offset), (w, y_center - offset), green, 2)
    results = hands.process(image)
    if results.multi_hand_landmarks:
        for idx, lm in enumerate(results.multi_hand_landmarks[0].landmark):
            if idx == 8:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 20, (255, 0, 255), cv2.FILLED)
                if cx >= (x_center + offset) and (y_center - offset) <= cy <= (y_center + offset):
                    x1_change = -snake_speed
                    y1_change = 0
                if cx <= (x_center - offset) and (y_center - offset) <= cy <= y_center + offset:
                    x1_change = snake_speed
                    y1_change = 0
                if cy <= (y_center - offset) and (x_center - offset) <= cx <= (x_center + offset):
                    y1_change = -snake_speed
                    x1_change = 0
                if cy >= (y_center + offset) and (x_center - offset) <= cx <= (x_center + offset):
                    y1_change = snake_speed
                    x1_change = 0
    image = cv2.flip(image, flipCode=1)

    return x1_change, y1_change, image

def game_loop():
    game_over = False
    game_close = False
 
    x1 = dis_width / 2
    y1 = dis_height / 2

    x1_change_prev = x1
    y1_change_prev = y1

    steps = 0

    x1_change = -1
    y1_change = 0
 
    snake_list = []
    length_of_snake = 1

    food_x = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
    food_y = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0


    def re_init_game():
        nonlocal x1, y1, x1_change, y1_change, snake_list, length_of_snake
        nonlocal food_x, food_y, x1_change_prev, y1_change_prev, steps
        x1 = dis_width / 2
        y1 = dis_height / 2

        x1_change = -1
        y1_change = 0

        x1_change_prev = x1
        y1_change_prev = y1

        steps = 0

        snake_list = []
        length_of_snake = 1

        food_x = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
        food_y = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0

    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)

    hands = mp_hands.Hands(
                model_complexity=0,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)

    while not game_over:
        while game_close:
            dis.fill(blue)
            message("You Lost! Press C-Play Again or Q-Quit", red)
            your_score(length_of_snake - 1)
            pygame.display.update()
 
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        re_init_game()
                        game_close = False
                        break
 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x1_change = -snake_speed
                    y1_change = 0
                elif event.key == pygame.K_RIGHT:
                    x1_change = snake_speed
                    y1_change = 0
                elif event.key == pygame.K_UP:
                    y1_change = -snake_speed
                    x1_change = 0
                elif event.key == pygame.K_DOWN:
                    y1_change = snake_speed
                    x1_change = 0

        cv_x1_change, cv_y1_change, image = cv2_finger_coord_image(hands, cap)

        x1_change = cv_x1_change if cv_x1_change != 0 else x1_change
        y1_change = cv_y1_change if cv_y1_change != 0 else y1_change

        dis.blit(cv2_image_to_surface(image), (0, 0))

        if x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0:
            game_close = True
        x1 += x1_change
        y1 += y1_change
        pygame.draw.rect(dis, green, [food_x, food_y, snake_block, snake_block])

        snake_head = [x1 + x1_change, y1 + y1_change]

        snake_head_prev = [x1, y1]
        if steps == 5:
            snake_list.append(snake_head_prev)
        if steps == 10:
            if len(snake_list) > length_of_snake:
                snake_list = snake_list[-length_of_snake:]
            steps = 0
        steps += 1


        for x in snake_list[:-1]:
            if x == snake_head:
                game_close = True
 
        our_snake(snake_head, snake_list)
        your_score(length_of_snake - 1)
 
        pygame.display.update()

        food_rect = pygame.Rect((food_x, food_y, snake_block, snake_block))
        head_rect = pygame.Rect((snake_head[0], snake_head[1], snake_block, snake_block))

        if food_rect.colliderect(head_rect):
            food_x = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
            food_y = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0
            length_of_snake += 1
 
        clock.tick(framerate)

    hands.close()
    cap.release()
    pygame.quit()
    quit()



game_loop()