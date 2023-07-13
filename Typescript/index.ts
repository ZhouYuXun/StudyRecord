//--------------------------基本類型----------------------------

let str:  string = "bruce"
let str1: string;
str1 = "bruce2"

let num: number = 1000
let boo: boolean = true
let n: null = null
let un: undefined = undefined

let test: any = true

//陣列

let arr: string[] = ['a', 'b']
let arr2: string[][] = [['aa', 'bb']]

//元祖

let tuple: [number, string, boolean] = [1, 'a', true]
let tuple2: [string, string][] = [['a', 'b']]

//--------------------------Enum枚舉----------------------------

enum LiveStatus{
    SUCCESS = 0,
    FAIL = -1,
    STREAMING = 1
}

const staus = LiveStatus.SUCCESS
console.log('staus', staus)

//--------------------------Union----------------------------

let aaa: number | string;
aaa = 100
aaa = 'str'

//--------------------------type----------------------------
//無法擴充

type A = number | string;
type B = boolean | string;
let a1: A
a1 = 999
a1 = 'str'

let b1: B
b1 = true

//--------------------------interface----------------------------
//可以擴充

interface User{
    name: string,
    age: number
}

//--------------------------object----------------------------

type Card = {
    name: string;
    desc: string;
}

interface Card2{
    name: string
    desc: string
}

interface Card2{
    age: number
    time?: number //可選
}

const obj: Card2 = {
    name: 'bruce',
    desc: '....',
    age: 100
}

//--------------------------function----------------------------

//參數類型

function hello(a: string, b: string){
    return a + b
}

function hello2(a: string, b: string): number{
    console.log(a + b)
    return 999
}

function hello3(a: number, b: boolean, c: string){
    return 100
}

function test2(a: number){
    console.log(a);
}

//undefined

function hello4(name: string, age?: number){
    if (age === undefined) return -1
        test2(age)
    return 
}

//箭頭涵式

const func = () => {}

const func2 = () => {
    return 1
}

//--------------------------斷言 unknown----------------------------

type Data = {
    "userId": number,
    "id": number,
    "title": string,
    "completed": boolean
}

async function getData() { 
    const res = await fetch('https://jsonplaceholder.typicode.com/todos/1')
    const data = await res.json() as Data
}


const data1: Data = {
    "userId": 1,
    "id": 1,
    "title": "delectus aut autem",
    "completed": false
}

type Bata = {
    name: string
}

//假設data1是動態資料
const bata = data1 as unknown as Bata

//--------------------------Class----------------------------

//public 公開
//private 私有
//protected 受保護

class Live {
    public roomName: string
    private id: string //只能防止同樣在寫程式碼的人使用
    protected name: string

    constructor (roomName1: string, id1: string, name1: string) {
        console.log('建立直播中')
        this.roomName = roomName1
        this.id = id1
        this.name = name1
    }
}

class CarLive extends Live{
    constructor (roomName1: string, id1: string, name1: string) {
        super (roomName1, id1, name1)
    }

    start(){
        super.roomName
        super.name
    }
}

//外面
const live = new Live('1號', '000001', 'bruce')
console.log('live', live)
const carLive = new CarLive('car room', '000002', 'bruce2')
console.log('carLive', carLive)

class Live2{
    //私有變數
    #name
    constructor(name: string){
        this.#name = name
    }
}

const live2 = new Live2('live2')
console.log('live2', live2)

interface CarProps{
    name: string
    age: number
    start: ()=>void
}

class Car implements CarProps{
    name: string;
    age: number;

    constructor (name: string, age: number){
        this.name = name
        this.age = age
    }

    start() {}

}

//--------------------------泛型----------------------------

function print<T> (data: T){
    console.log('data', data)
}

print<number>(999)
print<string>("string")
print<boolean>(true)

class Print<T> {
    data: T

    constructor(d: T){
        this.data = d
    }
}

const p = new Print<number>(999)
const p1 = new Print<string>('bruce')

console.log('p', p)
console.log('p1', p1)

//--------------------------utility----------------------------

interface CatInfo {
    age: number;
    breed: string;
  }
   
type CatName = "miffy" | "boris" | "mordred";

const cats: Record<CatName, CatInfo> = {
miffy: { age: 10, breed: "Persian" },
boris: { age: 5, breed: "Maine Coon" },
mordred: { age: 16, breed: "British Shorthair" },
};

cats.boris;

//Pick

// interface Todo {
// title: string;
// description: string;
// completed: boolean;
// }

// type TodoPreview = Pick<Todo, "title" | "completed">;

// const todo: TodoPreview = {
// title: "Clean room",
// completed: false,
// };

// todo;

  //Omit

  interface Todo {
    title: string;
    description: string;
    completed: boolean;
    createdAt: number;
  }
   
  type TodoPreview = Omit<Todo, "description">;
   
  const todo: TodoPreview = {
    title: "Clean room",
    completed: false,
    createdAt: 1615544252770,
  };
   
  todo;