import React, { useState } from 'react';
import logo from './logo.svg';
import './App.css';

// type TitleProps {
//   name: string
// }

//export
interface TitleProps {
  name: string
}

//import
interface TitleProps {
  desc?: string
}

const Title: React.FC<TitleProps> = ({name, desc}) => {
  return <p>{name}</p>
}

const App: React.FC = () => {
  const [title, setTitle] = useState('str')
  return (
    <div >
      <Title name="Bruce" desc='....'/>
    </div>
  );
}

export default App;
